'''Base class for finding birds in movies'''

import mpi4py.MPI as mpi
import numpy as np
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
        self.f_rate = int(self.cap.get(cv2.cv.CV_CAP_PROP_FPS))
        self.frames = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        self.f_sec = 1 / self.f_rate
        self.width = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

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
    
    def finialize(self):
        '''Closes everything'''
        self.cap.release()
        cv2.destroyAllWindows()
        out.release()

    def to_hhmmss(self, sec):
        min,sec = divmod(sec,60)
        hr,min = divmod(min,60)
        print "%d:%02d:%02d" % (hr,min,sec)
        return "%d:%02d:%02d" % (hr,min,sec)

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
        return range(frame_min, frame_max, stride)
        

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
            
    def camera(self, hr, min, sec):
        # Retrieve time stamps
        self.camera_time = subprocess.Popen("exiftool " + self.video_path + " | grep Date/Time\ Original | awk '{split($0,a,\" \"); print a[5]}' | awk '{split($0,a,\"+\"); print a[1]}'", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.camera_gmt = subprocess.Popen("exiftool " + self.video_path + " | grep Date/Time\ Original | awk '{split($0,a,\" \"); print a[5]}' | awk '{split($0,a,\"+\"); print a[2]}'", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.camera_time.stderr.read()#reads error output
        self.camera_time = self.camera_time.stdout.read()
        self.camera_gmt.stderr.read()#reads error output
        self.camera_gmt = self.camera_gmt.stdout.read()

        try:
            self.camera_time = time.strptime(self.camera_time, "%H:%M:%S")
        except ValueError, v:
            if len(v.args) > 0 and v.args[0].startswith('unconverted data remains: '):
                self.camera_time = self.camera_time[:-(len(v.args[0])-26)]
                self.camera_time = time.strptime(self.camera_time, "%H:%M:%S")

        try:
            self.camera_gmt = time.strptime(self.camera_gmt, "%H:%M")
        except ValueError, v:
            if len(v.args) > 0 and v.args[0].startswith('unconverted data remains: '):
                self.camera_gmt = self.camera_gmt[:-(len(v.args[0])-26)]
                self.camera_gmt = time.strptime(self.camera_gmt, "%H:%M")



        '''
        self.day_time = []
        for i_base in xrange(0,len(self.frame_time), self.size):
            i = i_base + self.rank
            if i < len(self.frame_time):
                self.day_time = datetime.timedelta(seconds=self.frame_time[i])
                '''
        print self.day_time

    def save_result(self, filename=None):
        '''
        
        Saves results to a .csv file  with 4 columns
        [actual_time, frame_num, frame_time (sec), frame_chi].
        
        If no files name given with be video_name.csv
        
        '''
        if filename is None:
            outfile = os.path.splitext(self.video_path)[0] + '.csv'
        else:
            outfile = os.path.splitext(filename)[0] + '.csv'

        if not hasattr(self, 'frame_no'):
            raise AttributeError('Must get reduced chi squared first')

        # Sort array
        self.frame_no.sort()

        out_array = np.vstack((self.frame_no, self.frame_time,
                               self.frame_chi)).T
        np.savetxt(outfile, out_array, delimiter=',',
                   header='day_time (HH:MM:SS), frame_number, frame_time (sec), reduced_chisquared')
        
    def plot(self, show=True):
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
        plt.savefig(self.video_path + ".jpg", dpi=self.dpi)
        if show:
            plt.show()
       
    def make_movie_mpi(self):
        '''
        TO CREATE MOVIE
        http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
        '''

        # Define the codec and create VideoWriter object
        #fourcc = cv2.cv.CV_FOURCC(*'XVID')
        fourcc = cv2.cv.CV_FOURCC('D','I','V','X')
        #fourcc = cv2.cv.CV_FOURCC('M','P','V','4')
        #fourcc = cv2.cv.CV_FOURCC('M','P','E','G')
        out = cv2.VideoWriter(self.video_path + '.avi',
                              fourcc, 50, (self.width,self.height))
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

    def make_movie(self):
        '''
        TO CREATE MOVIE
        http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
        '''

        # Define the codec and create VideoWriter object
        #fourcc = cv2.cv.CV_FOURCC(*'XVID')
        fourcc = cv2.cv.CV_FOURCC('D','I','V','X')
        #fourcc = cv2.cv.CV_FOURCC('M','P','V','4')
        #fourcc = cv2.cv.CV_FOURCC('M','P','E','G')
        out = cv2.VideoWriter(self.video_path + '.avi',fourcc, 50, (self.width,self.height))
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
                        
