from base import Video
import numpy as np
import cv2
import pylab as plt
from matplotlib.patches import Rectangle
import os
import cPickle as pik
from glob import glob
#import ipdb



class Obj_dectect(Video):
    '''User trains program to find object by selecting it'''

    def train(self, start_time=0, units='s'):
        '''Uses user input to train the program to find bird or other object.
        Makes folder with training info for later use'''
        print 'q or Esc key will quit, p to pause and start drawing, s to go forward 10 seconds and a to go back 10 sec'
        units = units.lower()
        if units == 's' or units == 'sec':
            start_from = self._sec2msec(start_time)
        elif units == 'min' or units == 'm':
            start_from = self._min2sec(start_time)
        self.cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, start_from)
        # Play
        draw = False
        while(True):
            # read frame-by-frame
            ret, frame = self.cap.read()
            if not ret:
                break
            cv2.imshow('frame', frame)
            key = cv2.waitKey(33)
            print key
            if key == ord('q') or key == ord('Q') or key == 27 or key == 1048689:
                break
            if key == ord('p') or key == 1048688:
                draw = True
                break
            if  key == ord('s') or key == 1048691:
                curframe = self.cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
                self.cap.set( cv2.cv.CV_CAP_PROP_POS_MSEC,
                              curframe + self._sec2msec(10))
            if  key == ord('a') or key == 1048673:
                curframe = self.cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
                self.cap.set( cv2.cv.CV_CAP_PROP_POS_MSEC,
                              curframe - self._sec2msec(10))
        if draw:
            # have user draw box around object for sampling
            cv2.destroyAllWindows()
            object_image = self.draw_box(frame)
            # identify keypoints
            self.get_keypoints(object_image)
            self.save_keypoints()
            
    def draw_box(self, frame):
        '''Draw box on image to get training object'''
        fig = plt.figure()
        plot = fig.add_subplot(111)
        plot.imshow(frame)
        #allow user to draw retangle
        draw = Draw_rec()
        plt.show()
        # Extract image and show
        return frame[draw.y0:draw.y1,draw.x0:draw.x1]

    def get_keypoints(self, image):
        '''Finds keypoints for object to find'''
        surf = cv2.SURF(2000)
        self.kp, self.des = surf.detectAndCompute(image, None)

    def save_keypoints(self,path='key'):
        '''saves keypoints for later use'''
        if not os.path.exists(path):
            os.mkdir(path)
        name = np.random.randint(9999)
        #pik.dump((self.kp,self.des), open(os.path.join(path,'%i.pik'%name)),2)
                 
    def load_keypoints(self, path='key'):
        '''loads keypoints'''
        files = glob(path,'*.pik')
        self.ks, self.des = [], []
        for i in files:
            ks, des = pik.load(open(i))
            self.ks = ks
            self.des = des

    def match_image(self, image, min_match=10):
        '''Detects object in image'''
        if not hasattr(self, 'flann'):
            # create matcher
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            self.flann = cv2.FlannBasedMatcher(index_params, search_params)
            self.surf = cv2.SURF(2000)
        # get descriports and keypoints
        kp_im, des_im = self.surf.detectAndCompute(image, None)
        matches = self.flann.knnMatch(self.des ,des_im, k=2)
        #count matches
        count = 0
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                count += 1
        return count
        

    def find_obj(self, max_frames=-1, fps=1, plot=False):
        '''Find object in a video'''
        if max_frames < 1:
            max_frames = self.frames
        # set number of steps to skip
        stride = int(round(self.f_rate / float(fps)))
        if not hasattr(self, 'kp'):
            # load keypoints
            self.load_keypoints()

        self.frame_no = []
        self.frame_chi = []
        self.frame_time = []
        for frame_no in xrange(0, max_frames, stride):
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_no)
            success, frame = self.cap.read()
            if not success:
                continue
            self.frame_no.append(frame_no)
            self.frame_time.append(self.cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
            self.frame_chi.append(self.match_image(frame))
            print self.frame_chi[-1],self._msec2min(self.frame_time[-1])
            if plot:
                # only plot if sucessfull
                if self.frame_chi[-1] > 10:
                    cv2.imshow('Got One',frame)
                else:
                    print 'Fail at %f'%frame_no

    def play_find(self, start_time=0, cascade_file='hornbil.xml'):
        '''uses cacade to find objects at the same time'''
        self.cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, start_time*10**3)
        hornbil = cv2.CascadeClassifier(cascade_file)
        for frame_no in range(1, self.frames):
            sucess, frame = self.cap.read()
            if not sucess:
                continue
            # check if there is a hornbil
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            birds = hornbil.detectMultiScale(gray, 1.02, 5)
            print self.cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)/(10**3)
            for (x,y,w,h) in birds:
                cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),2)
            cv2.imshow('frame', frame)
            cv2.waitKey(33)

class Draw_rec(object):
    def __init__(self):
        self.ax = plt.gca()
        self.rect = Rectangle((0,0), 1, 1)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        print 'press'
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release(self, event):
        print 'release'
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()
        #return self.x0,self.y0,self.x1,self.y1


class LetterStatModel(object):
    class_n = 26
    train_ratio = 0.5

    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

    def unroll_samples(self, samples):
        sample_n, var_n = samples.shape
        new_samples = np.zeros((sample_n * self.class_n, var_n+1), np.float32)
        new_samples[:,:-1] = np.repeat(samples, self.class_n, axis=0)
        new_samples[:,-1] = np.tile(np.arange(self.class_n), sample_n)
        return new_samples

    def unroll_responses(self, responses):
        sample_n = len(responses)
        new_responses = np.zeros(sample_n*self.class_n, np.int32)
        resp_idx = np.int32( responses + np.arange(sample_n)*self.class_n )
        new_responses[resp_idx] = 1
        return new_responses

class RTrees(LetterStatModel):
    def __init__(self):
        self.model = cv2.RTrees()

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        var_types = np.array([cv2.CV_VAR_NUMERICAL] * var_n + [cv2.CV_VAR_CATEGORICAL], np.uint8)
        #CvRTParams(10,10,0,false,15,0,true,4,100,0.01f,CV_TERMCRIT_ITER));
        params = dict(max_depth=10 )
        self.model.train(samples, cv2.CV_ROW_SAMPLE, responses, varType = var_types, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples] )


class KNearest(LetterStatModel):
    def __init__(self):
        self.model = cv2.KNearest()

    def train(self, samples, responses):
        self.model.train(samples, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.find_nearest(samples, k = 10)
        return results.ravel()


class Boost(LetterStatModel):
    def __init__(self):
        self.model = cv2.Boost()

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_samples = self.unroll_samples(samples)
        new_responses = self.unroll_responses(responses)
        var_types = np.array([cv2.CV_VAR_NUMERICAL] * var_n + [cv2.CV_VAR_CATEGORICAL, cv2.CV_VAR_CATEGORICAL], np.uint8)
        #CvBoostParams(CvBoost::REAL, 100, 0.95, 5, false, 0 )
        params = dict(max_depth=5) #, use_surrogates=False)
        self.model.train(new_samples, cv2.CV_ROW_SAMPLE, new_responses, varType = var_types, params=params)

    def predict(self, samples):
        new_samples = self.unroll_samples(samples)
        pred = np.array( [self.model.predict(s, returnSum = True) for s in new_samples] )
        pred = pred.reshape(-1, self.class_n).argmax(1)
        return pred


class SVM(LetterStatModel):
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples, responses):
        params = dict( kernel_type = cv2.SVM_LINEAR,
                       svm_type = cv2.SVM_C_SVC,
                       C = 1 )
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        return self.model.predict_all(samples).ravel()


class MLP(LetterStatModel):
    def __init__(self):
        self.model = cv2.ANN_MLP()

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_responses = self.unroll_responses(responses).reshape(-1, self.class_n)

        layer_sizes = np.int32([var_n, 100, 100, self.class_n])
        self.model.create(layer_sizes)

        # CvANN_MLP_TrainParams::BACKPROP,0.001
        params = dict( term_crit = (cv2.TERM_CRITERIA_COUNT, 300, 0.01),
                       train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,
                       bp_dw_scale = 0.001,
                       bp_moment_scale = 0.0 )
        self.model.train(samples, np.float32(new_responses), None, params = params)

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)
