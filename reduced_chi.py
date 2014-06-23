from base import Video
import numpy as np
import cv2
#import ipdb


class Reduice_chi(Video):


    def _get_inital_master(self, start, samples):
        '''samples previous frames from the video to get a inital master frame
        Start is the frame number to start, and samples is how many frames to
        sample'''
        # Get empty frame
        self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, start)
        sucess, inital_frame = self.cap.read()
        while not sucess:
            sucess, inital_frame = self.cap.read()
        inital_frame = np.float32(inital_frame)
        # get random samples
        sam_array = np.arange(start, start+samples)
        count = 0.
        for frame_no in sam_array:
            #self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_no)
            # get frame
            sucess, frame = self.cap.read()
            if not sucess:
               continue
            cv2.accumulateWeighted(np.float32(frame), inital_frame, 0.020)
            count += 1

        return inital_frame

    def parallel_moving_ave(self, frame_array, n_sample=120):
        '''Does same as moving_ave_chi but in parrallel'''
        # Check for master image
        if not 'moving_ave_frame' in vars(self):
            # Check if at zero
            if np.min(frame_array) - n_sample < 0:
                start = 0
            else:
                start = np.min(frame_array) - n_sample
            self.moving_ave_frame = self._get_inital_master(start, n_sample)

        self.frame_no = frame_array
        self.frame_chi = []
        self.frame_time = []
        # start chi square
        x,y,z = self.moving_ave_frame.shape
        ddof = float(x * y)
        for frame_no in frame_array:
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_no)
            success, frame = self.cap.read()
            if not success:
                # remove frame_no
                self.frame_no.pop(self.frame_no.index(frame_no))
                continue
            self.frame_time.append(self.cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
            # Cal chi
            frame = np.float32(frame)
            # add 1 so chi square can't be nan or inf
            self.frame_chi.append(chisquare(frame+1,
                                            self.moving_ave_frame+1).sum()
                                                                     / ddof)              
            # Cal new ave frame
            cv2.accumulateWeighted(frame, self.moving_ave_frame, 0.020)

        return self.frame_no, self.frame_time, self.frame_chi
        
    
    def moving_ave_chi(self, max_frames=-1, fps=1, plot=False):
        '''Uses moving average for master frame and calculates fps frames per second'''
        if max_frames < 1:
            max_frames = self.frames
        # set number of steps to skip
        stride = int(round(self.f_rate / float(fps)))
        # get inital aveage frame
        self.moving_ave_frame = self._get_inital_master(0, 120)
       
        # out puts
        self.frame_no = []
        self.frame_chi = []
        self.frame_time = []
        # start chi square
        x,y,z = self.moving_ave_frame.shape
        ddof = float(x * y)
        for frame_no in xrange(0, max_frames, stride):
            # Get chi square
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_no)
            success, frame = self.cap.read()
            if not success:
                continue
            self.frame_no.append(frame_no)
            self.frame_time.append(self.cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
            # Cal chi
            self.frame_chi.append(chisquare(frame, self.moving_ave_frame).sum()/ddof)
            # Cal new ave frame
            cv2.accumulateWeighted(frame, self.moving_ave_frame, 0.020)
            # Show moving Ave Frame and chi plot
            if plot:
                cv2.imshow('Ave Frame',self.moving_ave_frame)
                #cv2.waitKey(100)
    
def chisquare(f_obs, f_exp, axis=0):
    '''calculates chi squared test correctly for 3D arrays'''
    return (f_obs - f_exp)**2 / f_exp
