from image_diff.detectors import Background_detct
import mpi4py.MPI as mpi
import sys


video = Background_detct(sys.argv[1])
