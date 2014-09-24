import cv2
import numpy as np
import pylab as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import datetime



def plot(time, chi, filename=None, show=True, dpi=600):
    '''Makes plots of time vs chi'''
    majorLocator   = MultipleLocator(5)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator   = MultipleLocator(1)

    frameTime, frame_chi = time, chi
    
    dpi = 600
        
    '''minChi, maxChi = min(redChi), max(redChi)
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
            plt.show()'''

def make_movie(movie_obj):
        '''
        TO CREATE MOVIE
        http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
        '''
        
        # Define the codec and create VideoWriter object
        #fourcc = cv2.cv.CV_FOURCC(*'XVID')
        #fourcc = cv2.cv.CV_FOURCC('D','I','V','X')
        #fourcc = cv2.cv.CV_FOURCC('M','P','V','4')
        #fourcc = cv2.cv.CV_FOURCC('M','P','E','G')
        '''fourcc = cv2.cv.CV_FOURCC('X', '2', '6', '4')
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
                        out.write(frame)'''
                        
def get_frame(path, frame_no, show=False):
        '''gets frame from movie. frame_no should be a list'''
        assert isinstance(frame_no, (list, np.ndarray))
        cap = cv2.VideoCapture(path)
        out = []
        for frame in frame_no:
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame+25)
            sucess, image = cap.read()
            if not sucess:
                break
            out.append([image, frame])

        return out
