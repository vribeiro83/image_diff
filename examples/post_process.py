'''Does changepoint anaylisis to determine when and if a bird has arrive.
uses a HMM for change point. BIC to estimate the evidence for model selection.
Will try and Find video file and create a compilation video of birds.
Will put a blue "bird" when a new arrive happends.'''
import image_diff as id
import sys
from glob import glob
import mpi4py.MPI as mpi
import os

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
