from __init__ import Background_detct
import sys


video = Background_detct(sys.argv[1])
video.detect_movment(100)


