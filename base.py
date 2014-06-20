'''Base class for finding birds in movies'''

import mpi4py.MPI as mpi
import numpy as np
import cv2
import pylab as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import datetime as dt

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

    def finialize(self):
        '''Closes everything'''
        self.cap.release()
        cv2.destroyAllWindows()
        out.release()

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
            
    def save_result(self, filename=None):
        '''Saves results to a .csv file  with 3 colmns [frame_num,frame_time
        frame_chi].
        If no files name given with be video_name.csv'''
        if filename is None:
            outfile = os.path.splitext(self.video_path)[0] + '.csv'
        else:
            outfile = os.path.splitext(filename)[0] + '.csv'

        if not hasattr(self, 'frame_no'):
            raise AttributeError('Must get reduiced chi squared first')
        
        out_array = np.vstack((self.frame_no, self.frame_time,
                               self.frame_chi)).T
        np.savetxt(outfile, out_array, delimiter=',',
                   header='frame_number, frame_time, frame_chi')
        
    def plot(self, individual=True):
        print "I am about to start drawing those amazing figures you really want to see!"
        majorLocator   = MultipleLocator(5)
        majorFormatter = FormatStrFormatter('%d')
        minorLocator   = MultipleLocator(1)

        frame_no, frame_chi = self.frame_no, self.frame_chi
        # in mili seconds
        frameTime = np.asarray(self.frame_time)
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

        if not individual:
            if self.rank == 1:
                plt.plot(self._msec2min(frameTime),redChi, 'r-')
                plt.savefig(self.video_path + ".jpg", dpi=self.dpi)
                plt.clf()
        else:
            for i in xrange(len(frameTime)):
                pos = float(i)
                plt.axvline(x=pos)      
            plt.plot(self._msec2min(frameTime), redChi, 'k-')
            plt.savefig(self.video_path + ".jpg", dpi=self.dpi)
            plt.show()
            plt.clf()
        #plt.savefig(temp_folder + "reduced.eps")
        #plt.savefig(rootfile + ".eps")
        #ipdb.set_trace()
        #plt.show()
        '''
        for i in xrange(len(frameTime)):
            output = (temp_folder + "image%.7d.jpg" % i)

            ax = plt.subplot(1, 1, 1)

            plt.axhline(y=3*minChi)

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
            plt.text(xpos, ypos, rootfile)

            pos = float(i / 60.)
            #print output, pos
            plt.axvline(x=pos)
            plt.plot(frameTime, redChi, 'k-')
            plt.savefig(output)
            plt.clf()
            '''
    def make_movie(self):
        '''
    TO CREATE MOVIE
    http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    '''

        # Define the codec and create VideoWriter object
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        out = cv2.VideoWriter(self.video_path + '.avi',fourcc, 1, (self.width,self.height))
        if self.rank == 0:
            print "Making a movie"
            x_offset=y_offset= 0
            for i in xrange(0, len(self.frame_no), 1):
                #if self.frame_chi[i] > self.minChi:
                print self.frame_chi[i]
                    #images = cv2.imread(files[i])
                    #s_img = cv2.imread(output[i])
                    #s_img = cv2.resize(s_img, (0,0), fx=0.5, fy=0.5) 
                    #images[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
                    #out.write()
