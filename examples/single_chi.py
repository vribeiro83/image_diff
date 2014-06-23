from __init__ import Reduice_chi
import sys

video = Reduice_chi(sys.argv[1])
video.moving_ave_chi()
video.plot()
video.save_result()
