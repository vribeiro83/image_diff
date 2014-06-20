#/usr/bin/python2

'''
time mpiexec -n 8 python batch.py movie_filename
'''

from reduced_chi import Reduice_chi
from background_subtract import Background_detct

__all__ = ['reduced_chi','background_subtract']

# put help message
