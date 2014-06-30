#!/usr/bin/python2

from reduced_chi import Reduced_chi
from background_subtract import Background_detect
from object_detect import Obj_dectect

__all__ = ['reduced_chi','background_subtract','object_detect']

# put help message
'''
time mpiexec -n 8 python batch.py movie_filename
'''
