'''Does changepoint anaylisis to determine when and if a bird has arrive.
uses a HMM for change point. BIC to estimate the evidence for model selection.
Will try and Find video file and create a compilation video of birds.
Will put a blue "bird" when a new arrive happends.'''
import image_diff as id
import sys
from glob import glob
import mpi4py.MPI as mpi
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def find_csv(path):
    '''Returns list of .csv. dict where keys are dir and items is path to csv'''
    files = {}
    walk = os.walk(path)
    #walk through dir structure finding all csv files
    for dirpath, dirnames, filenames in walk:
        for file in filenames:
            if file.find('.csv') >= 0:
                # check if key exsists
                if not os.path.split(dirpath)[-1] in files:
                    files[os.path.split(dirpath)[-1]] = []
                files[os.path.split(dirpath)[-1]].append(
                    os.path.join(dirpath,file))
    return files

def make_plots(path, dir_plot=True):
    '''Makes chi squared plots for all .csv without .jpgs. dir_plot
    makes a plot of all csv files in the dir'''
    files = find_csv(path)
    for dir in files:
        # open csv
        temp_file = np.loadtxt()
    
def find_hornbils(path, evidence=6., doub_check=False):
    '''Makes videos for each dir of finding birds using change point analyisis.
     Will put a big blue label up between events.
     Uses BIC criteron to decide if bird or background noise.
     If want to double check if bird is there, use cascade classfiyer and
     put path to .xml file.

     Uses multiprocessing.'''
    files = find_csv(path)
    for dir in files:

def header(path):
    '''Fix header so no commas and tab seperated'''
    with open(path,'r+') as f:
        #check format
        line = f.readline()
        #check for inital \t and remove it
        if line.find('\t') == 0:
            f.seek(0)
            f.write(line[1:-1])
        if len(line.split(',')) == 1:
            return None
        line = line.split(',')[0][1:] + '\t' +line.split(',')[1][1:-1]
        f.seek(0)
        f.write(line)
        
        
def test_evidence(path, video_path):
    '''Does test to see what bayes factor is needed to find a bird versus noise'''
    import pylab as lab
    lab.ion()
    # get csv files
    files = find_csv(path)
    files = np.concatenate(files.values())
    # make dlik vs percent bird list
    birdVlik = []
    # dlik, raveled image
    training = []
    # true or false (1,0)
    has_bird = []

    for file in np.random.choice(files, size=50, replace=False):
        # samplies a lot of videos
        # check if video exsists
        vid_path = os.path.join(video_path,file[file.find("HB"):-3] +'MTS')
        if not os.path.exists(vid_path):
            continue
        # fix csv header
        header(file)
        # get delta log-like
        data = pd.DataFrame.from_csv(file, infer_datetime_format=True, sep='\t')
        chi_name = data.columns[-1]
        dlik, chpt = id.utils.find_changepoints(data[chi_name])
        # get frames from high states
        low, high = np.unique(chpt)
        index = np.where(chpt == high)[0]
        img = id.utils.av.get_frame(vid_path,
                                    np.asarray(data[u'frame_number'][index]))
        max = 10
        for j, i in enumerate(img):
            training.append([dlik, np.ravel(i[0])])
            lab.imshow(i[0])
            lab.title(str(i[1]))
            lab.show()
            answer = raw_input('Has bird? (y/n)')
            if answer.lower() in ['y', 'yes']:
                has_bird.append(1)
            else:
                has_bird.append(0)
            lab.close('all')
            if j > max:
                break
        birdVlik.append([np.sum(has_bird)/ float(len(has_bird)), dlik])
        # do some prediction modeling to figure out evidence needed
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(training, has_bird)
