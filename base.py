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
        # Create datetime array
        self.frame_datetime = np.asarray(self._time2datetime(self.frame_time))
        #save as a dataframe since they can have multiple types
        out_data = np.vstack((self.frame_no, self.frame_time,
                               self.frame_chi)).T
        # rm nan and inf values
        out_data = out_data[np.isfinite(out_data[:,2])]
        self.frame_no = out_data[np.isfinite(out_data[:,2]), 0]
        self.frame_time = out_data[np.isfinite(out_data[:,2]),1]
        self.frame_chi = out_data[np.isfinite(out_data[:,2]),2]
        self.frame_datetime = self.frame_datetime[np.isfinite(out_data[:,2])]
        # get DataFrame ready
        temp = pd.DataFrame(data=out_data, index= self.frame_datetime,
                             columns=['Date_time', 'frame_number',
                                    'frame_time_(msec)','reduced_chisquared'])
        temp.to_csv(outfile, sep="\t")
        
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
                        
def Sigmaclip(array, low=4., high=4, axis=None):
    '''(ndarray, int, int, int) -> ndarray

    Iterative sigma-clipping of along the given axis.
	   
    The output array contains only those elements of the input array `c`
    that satisfy the conditions ::
	   
    mean(c) - std(c)*low < c < mean(c) + std(c)*high
	
    Parameters
    ----------
    a : array_like
    data array
    low : float
    lower bound factor of sigma clipping
    high : float
    upper bound factor of sigma clipping
    
    Returns
    -------
    c : array
    sigma clipped mean along axis
    '''
    c = np.asarray(array)
    if axis is None or c.ndim == 1:
        from scipy.stats import sigmaclip
        out = sigmaclip(c)[0]
        return out.mean(), out.std()
    #create masked array
    c_mask = np.ma.masked_array(c, np.isnan(c))
    delta = 1
    while delta:
           c_std = c_mask.std(axis=axis)
           c_mean = c_mask.mean(axis=axis)
           size = c_mask.mask.sum()
           critlower = c_mean - c_std*low
           critupper = c_mean + c_std*high
           indexer = [slice(None)] * c.ndim
           for i in xrange(c.shape[axis]):
               indexer[axis] = slice(i,i+1)
               c_mask[indexer].mask = np.logical_and(
                   c_mask[indexer].squeeze() > critlower, 
                   c_mask[indexer].squeeze() < critupper) == False
           delta = size - c_mask.mask.sum()
    return c_mask.mean(axis).data, c_mask.std(axis).data

def find_changepoints(y, zmax=20):
    '''Finds changepoints in data with hmm and viterbi algo.
    Estimates the transmission and emission probs from data'''
    y = np.asarray(y)
    # estimate transmission pdf as exp with boot straping?
    randomsample = np.random.choice(y, size=len(y)*20)
    pdf_y,pdf_x = np.histogram(np.abs(np.diff(randomsample)), 400, normed=True)
    pdf_y[pdf_y == 0] = 1e-99
    pdf_x = pdf_x[:-1]
    trans_p = General_PDF(pdf_x, pdf_y)
    # Get emmission probs by sigma clipping data
    emiss_mean, emiss_std = Sigmaclip(y)
    states = np.histogram(np.abs(np.diff(randomsample)), 50)[1]
    viterbi(y, states, norm(loc=emiss_mean, scale=emiss_std), trans_p, norm)
    
class General_PDF(object):
    def __init__(self, x, y):
        self.x = x
        # renormalize
        self.y = y/y.sum()
    def __call__(self, state1, state2):
        '''Calulates the prob of abs(state1 - state2) from input data'''
        return np.interp(np.abs(state1 - state2), self.x, self.y)
    def pdf(self, state1, state2):
        return self.__call__(state1, state2)
    def logpdf(self, state1, state2):
        return np.log(self.__call__(state1, state2))

def viterbi(obs, states, start_p, trans_p, emit_p):
    '''Viterbi algo from http://en.wikipedia.org/wiki/Viterbi_algorithm'''
    V = [{}]
    path = {}
    states = np.sort(states)
    # Initialize base cases (t == 0)   
    for y in states:
        V[0][y] = start_p.logpdf(y) + emit_p.logpdf(obs[0], y, start_p.std()) 
        path[y] = [y]
 
    # Run Viterbi for t > 0
    for t in xrange(1, len(obs)):
        #print t
        V.append({})
        newpath = {}
        # stuff for loop
        argsort = np.argsort(V[t-1].keys())
        oldV = np.asarray(V[t-1].values())[argsort]
        Emit = emit_p.logpdf(obs[t], states, start_p.std())
        for y in states:
            probs  = (oldV + trans_p.logpdf(states, y) + Emit)
            V[t][y] = probs.max()
            newpath[y] = path[states[probs.argmax()]] + [y]
 
        # Don't need to remember the old paths
        path = newpath
    n = 0           # if only one element is observed max is sought in the initialization values
    if len(obs)!=1:
        n = t
    (prob, state) = max((V[n][y], y) for y in states)
    return (prob, path[state])
