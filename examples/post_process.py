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
import scipy.stats as stats


def find_csv(path):
    '''Returns list of .csv. dict where keys are dir and items is path to csv'''
    files = {}
    walk = os.walk(os.path.abspath(path))
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

def mkpath(file, out_path, start='HB'):
    '''Makes dir tree for file starting in file's start in out_path.'''
    index = file.find(start)
    if index < 0:
        return None
    make_path = os.path.split(os.path.join(out_path, file[index:]))[0].split(os.path.sep)
    for i in range(1, len(make_path)):
        if not os.path.exists(os.path.sep.join(make_path[:i])):
            os.mkdir(os.path.sep.join(make_path[:i]))
    # make last dir
    if not os.path.exists(os.path.sep.join(make_path)):
        os.mkdir(os.path.sep.join(make_path))
    return os.path.sep.join(make_path)
    
def find_hornbils_chi(path, out_path, p_value=0.01, double_check=False):
    '''Combines all csv files into single file with columns of:
    chi_max, duration, video time (min), frame_no, path to .mts file
    path - path to csv dirs
    out_path - makes same dir tree as path but with 1 csv file in each leaf
    p_value - p_value to use to find birds (affects FPR and FNR)
    double_check - if true, will use a classifyer to check if bird is there.'''
    # make dir for out_path if not exsists
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    files = find_csv(path)
    tot_files = len(files.values())
    i = 0
    for dir in files.keys()[::-1]:
        print ' %i directories left'%(tot_files - i)
        i += 1 
        # make path if no there
        write_path = mkpath(files[dir][0], out_path)
        # process all files in dir
        data, chpt, dir_list = [], [], []
        for csv in files[dir]:
            header(csv)
            # get delta log-like
            try:
                data.append(pd.DataFrame.from_csv(csv, infer_datetime_format=True, sep='\t'))
                chi_name = data[-1].columns[-1]
                time_name = data[-1].columns[1]
                # find birds with chi test
                temp_chpt = chi_test(data[-1][chi_name], p_value)[1]
                # check if any chpts
                if len(np.unique(temp_chpt)) == 1:
                    data.pop(-1)
                    continue
                chpt.append(chpt_details(temp_chpt,data[-1][time_name],
                                        data[-1][chi_name]))
                dir_list.append([csv]*chpt[-1][0].shape[0])
            except:
                print csv
                raise
        # check if anything was found
        if len(dir_list) == 0:
            continue
        dir_list = np.hstack(dir_list)
        to_write = np.vstack((np.hstack(chpt),
                               dir_list.reshape(1, dir_list.shape[0]))).T
        # remove nans
        index = np.isfinite(np.float32(to_write[:,1]))
        to_write = to_write[index]
        # change time
        to_write[:,1] = map(hms, to_write[:,1])
        # change and remove events with duration less than 0.5 sec
        to_write[:,2] = np.round(np.float32(to_write[:,2]) * 10**-3,3)
        to_write = to_write[np.float32(to_write[:,2]) > 0.5]
        # change original file from .csv to .MTS
        to_write[:,4] = [j[:-3]+'MTS' for j in to_write[:,4]]
        # write file
        np.savetxt(os.path.join(write_path,'all.csv'), to_write,'%s', delimiter=','
                   ,header='Frame_number, time(HH:MM:SS), Duration (sec), max_chi, original_File' )
            
                
 
    
def chpt_details(chpt, time, chi):
    '''Returns frames, mean time(msc), duration,  mean frame of a change point'''
    # make arrays
    time = np.asarray(time)
    chi = np.asarray(chi)
    chpt = np.asarray(chpt)
    if len( np.unique(chpt)) == 2:
        low, high = np.unique(chpt)
    else:
        return [],[],[]
    index = np.where(chpt == high)[0]
    out_index = []
    out_max_chi = []
    out_duration = []
    out_time = []
    start = 0
    for diff in np.where(np.diff(index) > 1)[0]:
        # print chpt[index[start]:index[diff]+1]
        out_index.append((index[start]+index[diff]+1)/2)
        out_max_chi.append(np.max(chi[start:diff+1]))
        out_duration.append(time[diff+1] - time[start])
        out_time.append(np.mean(time[start:diff+1]))
        start = diff +1
    return np.asarray(out_index), np.asarray(out_time),np.asarray(out_duration), np.asarray(out_max_chi)


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
        
def chi_test(chi, p_value=.01):
    '''Does chi test with 1 ddof to see if change was significant'''
    p = stats.chisqprob(np.asarray(chi), 1)
    return p.max(), np.int32(p <= p_value)
            
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
        time_name = data.columns[1]
        # HMM 93% FPR
        #dlik, chpt = id.utils.find_changepoints(data[chi_name])
        # chi values 96% FPR
        dlik, chpt = chi_test(data[chi_name])
        # get frames, max chi and durration from high states
        index, durration, max_chi = chpt_details(chpt, data[time_name]
                                                 ,data[chi_name])
        # get images for testing        
        img = id.utils.av.get_frame(vid_path,
                                    np.asarray(data[u'frame_number'][index]))
        max = 10
        for j, i in enumerate(img):
            training.append([dlik, durration[j], max_chi[j]])
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
        # do some prediction modeling to figure out evidence needed
    clf = RandomForestClassifier(n_estimators=10)
    # put into correct format
    # some arrays have different size remove?
    size = np.asarray([len(j) for i,j in training])
    max_size = 0
    for i in np.unique(size):
        if sum(size == i) > max_size:
            max_size = sum(size == i)
            out_size = i +0
    # do it in a weird way to save memory
    tra = []
    img = []
    lik = []
    has_bird = np.asarray(has_bird)[np.where(size == out_size)[0]]
    for i in np.where(size == out_size)[0]:
        lik.append([training[i][0]])
        img.append(training[i][1])
    img = np.vstack(img)
    #tra = np.hstack((lik, img))
    clf.fit(lik, has_bird)


def hms(msec):
    '''Turns msec to hh:mm:ss str format'''
    sec = float(msec) * 10**-3
    m = int(sec / 60)
    sec = sec % 60
    h = int(m / 60)
    m = m % 60
    return '%02i:%02i:%02i'%(h,m,sec)


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    out_path = sys.argv[2]
    find_hornbils_chi('../honrbil/', 'test')
