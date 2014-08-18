from image_diff.detectors import Reduced_chi
import sys

video = Reduced_chi(sys.argv[1])
video.moving_ave_chi()
video.save_result()
video.plot()

