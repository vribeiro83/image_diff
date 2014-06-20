from ../__init__ import Background_detct
import mpi4py.MPI as mpi
import sys


video = Background_detct(sys.argv[1])
