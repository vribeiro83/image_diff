'''Base class for finding birds in movies'''

import mpi4py.MPI as mpi
import numpy as np
import pandas as pd
from scipy.stats import norm
import cv2
import pylab as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import os
import datetime
import time
import subprocess

class Video(object):
    
    def __init__(self, video_path):
        self.comm = mpi.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        #video info
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not np.isfinite(self.cap.get(cv2.cv.CV_CAP_PROP_FPS)):
            self.f_rate = self._man_fps()
        else:
            self.f_rate = int(self.cap.get(cv2.cv.CV_CAP_PROP_FPS))
        self.frames = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        if self.frames < 0:
            # there is a problem get frame num from iterating
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, abs(self.frames))
            if not self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) ==abs(self.frames):
                raise ValueError('Problem with video ')
            sucess, frame = self.cap.read()
            if sucess:
                sucess = False
                self.frames = abs(self.frames)
            else:
                sucess = True
                self.frames = 0
            while sucess:
                self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,self.frames)
                sucess, frame = self.cap.read()
                self.frames += 500
                
            print 'frames =', self.frames
        self.f_sec = 1 / self.f_rate
        self.width = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        
    def _man_fps(self):
        '''manually calculate fps from 2 different frame times'''
        t_start = self.cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
        self.cap.read()
        t_stop = self.cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
        return 1. /self._msec2sec(t_stop - t_start)
           
    def _msec2min(self, msec):
        '''Converts miliseconds to min'''
        return msec * 1.6666666666666667e-05
    
    def _msec2sec(self, msec):
        '''Converts mili-sec to sec'''
        return msec * 10**-3

    def _sec2msec(self, sec):
        '''Converts seconds to millisec'''
        return sec * 10.**3

    def _min2msec(self, min):
        '''Converts minues to millisec'''
        return min/1.6666666666666667e-05

    def _time2datetime(self, t):
        '''
        Turns frame time into datetime (YYYY:MM:DD HH:MM:SS) from video
        header. Assumes t is in milliseconds from camera_time
        '''
        if not hasattr(self, 'camera_time'):
            self.get_camera_time()
        if isinstance(t, (list, np.ndarray)):
            out = [self.camera_time + datetime.timedelta(milliseconds=i)
                   for i in t]
        else:
            out = self.camera_time + datetime.timedelta(milliseconds(t))
        return out
                
    
    def finialize(self):
        '''Closes everything'''
        self.cap.release()
        cv2.destroyAllWindows()
        out.release()



    def play(self, start_frame=0):
        '''Plays video using opencv2'''
        print 'q key will quit, r key will return frame number'
        self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, start_frame)
        while(True):
            # read frame-by-frame
            ret, frame = self.cap.read()
            if not ret:
                break
            cv2.imshow('frame', frame)
            key = cv2.waitKey(33)
            #print key
            if key == ord('q') or key == ord('Q') or key == 27 or key == 1048689:
                break
            if key == ord('r') or key == ord('R') or key == 1048690:
                print 'frame number is %i'%self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        # When everything done, release windowa
        cv2.destroyAllWindows()

    def mpi_work_split(self, max_frames=-1, size=None, fps=1, start=0):
        '''Splits work up from worker size'''
        if not size is None and size > 1:
            self.size = size
        if max_frames < 1:
            max_frames = self.frames
        
        stride = int(round(self.f_rate / float(fps)))
        # Make split array
        split = np.linspace(start, max_frames, self.size+1).round()
        frame_min, frame_max = int(split[self.rank]),int(split[self.rank + 1])
        return xrange(frame_min, frame_max, stride)
        

    def aggrigate_to_root(self, frame_no, frame_time, frame_chi):
        '''Send and organizes data at root'''
        if self.rank == 0:
            reciv =[(frame_no, frame_time, frame_chi)]
            for source in xrange(1,self.size):
                reciv.append(self.comm.recv(source=source))
                print 'Recived from process %i'%source
            self.frame_no, self.frame_time, self.frame_chi = np.hstack(reciv)

        else:
            self.comm.send((frame_no, frame_time, frame_chi), dest=0)

    def get_camera_time(self, tmzone='02:00'):
        '''
        Retrieve camera information using exiftool on command line
        This does not yet retrieve timezone need some working
        '''
        # Retrieve time stamps
        camera_time = subprocess.Popen("exiftool " + self.video_path +
                                        " | grep Date/Time\ Original",
                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        error = camera_time.stderr.read()
        if len(error):
            # Problem with reading
            raise OSError(error)
        unproc_str = camera_time.stdout.read()
        unproc_str = unproc_str[unproc_str.find(':')+1:]
        unproc_str = unproc_str[:-7]
        # get date, time and time zone
        # this does not actually get the time zone
        self.camera_time = datetime.datetime.strptime(unproc_str,
                                        ' %Y:%m:%d %H:%M:%S')


    def save_result(self, filename=None):
        '''
        
        Saves results to a .csv file  with 5 columns
        [Date, actual_time, frame_num, frame_time (sec), reduced chisquared].
        
        If no files name given with be video_name.csv
        
        '''
        if filename is None:
            outfile = os.path.splitext(self.video_path)[0] + '.csv'
        else:
            outfile = os.path.splitext(filename)[0] + '.csv'

        if not hasattr(self, 'frame_no'):
            raise AttributeError('Must get reduced chi squared first')

        # Sort array
        argsort = np.argsort(self.frame_no)
        self.frame_no = np.sort(self.frame_no)
        self.frame_time = np.asarray(self.frame_time)[argsort]
        self.frame_chi = np.asarray(self.frame_chi)[argsort]
        # rm nan and inf
        argnan = np.isfinite(np.float32(self.frame_chi))
        # Create datetime array
        self.frame_datetime = np.asarray(self._time2datetime(self.frame_time))
        #save as a dataframe since they can have multiple types
        out_data = np.vstack(( self.frame_no[argnan], 
                              self.frame_time[argnan], self.frame_chi[argnan] )).T
        # rm nan and inf values
        # get DataFrame ready
        temp = pd.DataFrame(data=out_data, index= self.frame_datetime[argnan],
                             columns=['frame_number',
                                    'frame_time_(msec)','reduced_chisquared'])
        temp.to_csv(outfile, sep="\t",index_label='Date_time')
        
    def plot(self, filename=None, show=True):
        print "I am about to start drawing those amazing figures you really want to see!"
        majorLocator   = MultipleLocator(5)
        majorFormatter = FormatStrFormatter('%d')
        minorLocator   = MultipleLocator(1)

        frame_no, frame_chi = np.copy(self.frame_no), np.copy(self.frame_chi)
        # in mili seconds
        frameTime = np.copy(self.frame_time)
        redChi = frame_chi

        self.dpi = 600
        # time in min
        #ipdb.set_trace()
        minTime, maxTime = self._msec2min(min(frameTime)), self._msec2min(max(frameTime))
        minChi, maxChi = min(redChi), max(redChi)
        averageChi = float(sum(redChi) / len(redChi))

        #plt.axhline(y=3*minChi)

        ax = plt.subplot(1, 1, 1)
        #plt.figure()
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_major_formatter(majorFormatter)
        ax.xaxis.set_minor_locator(minorLocator)
    
        plt.tick_params(axis='both', which='major', length=10)
        plt.tick_params(axis='both', which='minor', length=5)
        
        plt.ylim([minChi, maxChi])
        plt.xlim([minTime, maxTime])
        plt.xlabel("Time (min)")
        plt.ylabel("Reduced Chi-squared")

        plt.gca().set_yscale('log')

        xpos, ypos = float(minTime), float(maxChi/1.5)
        plt.text(xpos, ypos, self.video_path)

 
        plt.plot(self._msec2min(frameTime), redChi, 'k-')
        if filename is None:
            plt.savefig(self.video_path + ".jpg", dpi=self.dpi)
        else:
            plt.savefig(filename + ".jpg", dpi=self.dpi)
        if show:
            plt.show()
       
    def make_movie_mpi(self):
        '''
        TO CREATE MOVIE
        http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
        '''

        # Define the codec and create VideoWriter object
        #fourcc = cv2.cv.CV_FOURCC(*'XVID')
        #fourcc = cv2.cv.CV_FOURCC('D','I','V','X')
        #fourcc = cv2.cv.CV_FOURCC('M','P','V','4')
        #fourcc = cv2.cv.CV_FOURCC('M','P','E','G')
        fourcc = cv2.cv.CV_FOURCC('L','M','P','4')
        out = cv2.VideoWriter(self.video_path + '.avi',
                              fourcc, self.f_rate, (self.width/4,self.height/4))
        print "Making a movie, I am rank: ", self.rank
        for i_base in range(0, len(self.frame_no)-1, self.size):
            i = i_base + self.rank
            if i < len(self.frame_no):
            #print self.frame_chi[j]
                if i == len(self.frame_no) and self.frame_chi[i] > 3.:
                    print "Oh be a fine girl kiss me now"
                elif self.frame_chi[i] > 3. and self.frame_chi[i+1] > 3.:
                    print "it's true", self.frame_no[i], self.rank, len(self.frame_no)

                    for j in range(int(self.frame_no[i]), int(self.frame_no[i+1]), 1):
                        self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, j)
                        success, frame = self.cap.read()
                        out.write(frame)
                        
                elif self.frame_chi[i] > 3. and self.frame_chi[i+1] < 3.:
                    print "not true", self.frame_no[i], self.rank, len(self.frame_no)
                    for j in range(int(self.frame_no[i]), int(self.frame_no[i+1]), 1):
                        self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, j)
                        success, frame = self.cap.read()
                        out.write(frame)

    def make_movie(self, movie_obj):
        '''
        TO CREATE MOVIE
        http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
        '''
        
        # Define the codec and create VideoWriter object
        #fourcc = cv2.cv.CV_FOURCC(*'XVID')
        #fourcc = cv2.cv.CV_FOURCC('D','I','V','X')
        #fourcc = cv2.cv.CV_FOURCC('M','P','V','4')
        #fourcc = cv2.cv.CV_FOURCC('M','P','E','G')
        fourcc = cv2.cv.CV_FOURCC('X', '2', '6', '4')
        out = cv2.VideoWriter(self.video_path + '.mp4',fourcc, self.fps,
                              (self.width/4,self.height/4))
        print "Making a movie, I am rank: ", self.rank
        for i in range(len(self.frame_no)):
            if i < len(self.frame_no):
            
                if i == len(self.frame_no) and self.frame_chi[i] > 3.:
                    print "Oh be a fine girl kiss me now"
                elif self.frame_chi[i] > 3. and self.frame_chi[i+1] > 3.:
                    print "it's true", self.frame_no[i], self.rank, len(self.frame_no)

                    for j in range(int(self.frame_no[i]), int(self.frame_no[i+1]), 1):
                        self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, j)
                        success, frame = self.cap.read()
                        out.write(frame)
                elif self.frame_chi[i] > 3. and self.frame_chi[i+1] < 3.:
                    print "not true", self.frame_no[i], self.rank, len(self.frame_no)
                
                    for j in range(int(self.frame_no[i]), int(self.frame_no[i+1]), 1):
                        self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, j)
                        success, frame = self.cap.read()
                        out.write(frame)
                        
