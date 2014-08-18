#!/usr/bin/python2
from image_diff.detectors import Reduced_chi
import mpi4py.MPI as mpi
import sys

video = Reduced_chi(sys.argv[1])
# time run
t_multi = mpi.Wtime()
work_array = video.mpi_work_split()
frame_no, frame_time, frame_chi = video.parallel_moving_ave(work_array)
video.aggrigate_to_root(frame_no, frame_time, frame_chi)
video.comm.barrier()

t_multi -= mpi.Wtime()
if video.rank == 0:
    # Make plot
    video.save_result()
    video.plot(show=False)
    #video.make_movie()
