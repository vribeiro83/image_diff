from image_diff.detectors import Reduced_chi
import mpi4py.MPI as mpi
import sys
import os
from glob import glob
import time
import shutil 

def num_files_left(files):
    '''Returns number of files to be completed'''
    rem_file = 0
    for f in files:
        local = '/'.join(f.split('/')[4:])
        #if os.path.exists(local+'.jpg') and os.path.exists(local[:-4] +'.csv'):
        if os.path.exists(local[:-4] +'.csv'):
            rem_file += 1
    return len(files) - rem_file

files = glob(os.path.join(sys.argv[1],'HB*/*/*/*.MTS'))
for File in files:
    # Check if path exsists
    local = '/'.join(File.split('/')[4:])
    #local = '/'.join(File.split('/')[1:])
    #if os.path.exists(local+'.jpg') and os.path.exists(local[:-4] +'.csv'):
    if os.path.exists(local[:-4] +'.csv'):
        continue
    else:
        if mpi.COMM_WORLD.rank == 0:
            # make path if not there to save point
            for i in range(1,len(local.split('/'))):
                if not os.path.exists('/'.join(local.split('/')[:i])):
                    os.mkdir('/'.join(local.split('/')[:i]))
    # Run if more than 1 chain
    if mpi.COMM_WORLD.Get_size() > 1:
        if mpi.COMM_WORLD.rank == 0:
            print File, '\n%i files are left.'%num_files_left(files)
            # copy file to local
            #shutil.copy(File, 'temp.MTS')
            time_start = time.time()
        try:
            mpi.COMM_WORLD.barrier()
            #video = Reduced_chi('temp.MTS')
            video = Reduced_chi(File)
        except ValueError:
            print 'There was a problem with %s.'%File
            continue
        work_array = video.mpi_work_split()
        frame_no, frame_time, frame_chi = video.parallel_moving_ave(work_array)
        video.aggrigate_to_root(frame_no, frame_time, frame_chi)
        if video.rank == 0:
            video.save_result(local[:-4] +'.csv')
            # Make plot
            #video.plot(local[:-4], show=False)
        video.comm.barrier()
        if mpi.COMM_WORLD.rank == 0:
            print 'It took %.1f seconds for a %2.1f GB file'%(time.time() - time_start, os.path.getsize(File)*10**-9)
    else:
        # Print missing
        print '%s is missing.'%File
        print '\nFiles left is %i'%num_files_left(files)        
        
