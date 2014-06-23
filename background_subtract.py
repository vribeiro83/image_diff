from base import Video
import cv2
import numpy as np
import ipdb


class Background_detct(Video):

    def detect_movment(self, max_frames=-1, fps=1, plot=False):
        '''Does background modeling to find movment in movie'''
        if max_frames < 1:
            max_frames = self.frames
        # set number of steps to skip
        stride = int(round(self.f_rate / float(fps)))
        # set output arrays
        self.frame_no = []
        self.frame_chi = []
        self.frame_time = []
        # background model
        fgbg = cv2.BackgroundSubtractorMOG()
        for frame_no in xrange(0, max_frames, stride):
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_no)
            success, frame = self.cap.read()
            if not success:
                continue

            self.frame_no.append(frame_no)
            self.frame_time.append(self.cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
            # Calculate difference
            fgmask = fgbg.apply(frame)
            # smooth
            # find all movements
            # check if matches bird profile
            self.frame_chi.append(float(fgmask.sum()))
            if plot:
                cv2.imshow('frame',fgmask)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
            #ipdb.set_trace()
