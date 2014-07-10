from __init__ import Reduced_chi
import mpi4py.MPI as mpi
import sys
import os
from glob import glob


files = glob(os.path.join(sys.argv[1],'HB*/*/*/*.MTS'))
for File in files:
    # Check if path exsists
    local = '/'.join(file.split('/')[2:])
    if os.path.exists(local+'.jpg') and os.path.exists(local[:-4] +'.csv'):
        continue
    else:
        # make path if not there to save point
        for i in range(1,len(local.split('/'))):
            if not os.path.exists('/'.join(local.split('/')[:i])):
                os.mkdir('/'.join(local.split('/')[:i]))
    # Run if more than 1 chain
    if mpi.COMM_WORLD.Get_size() > 1:
        video = Reduced_chi(File)
        work_array = video.mpi_work_split()
        frame_no, frame_time, frame_chi = video.parallel_moving_ave(work_array)
        video.aggrigate_to_root(frame_no, frame_time, frame_chi)
        video.comm.barrier()
        if video.rank == 0:
            video.save_result(local[:-4] +'.csv')
            # Make plot
            video.plot(local[:-4], show=False)
        video.comm.barrier()
    else:
        # Print missing
        print '%s is missing.'%File
                    
