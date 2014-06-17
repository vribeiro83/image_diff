#/usr/bin/python

'''
time mpiexec -n 8 python batch.py movie_filename
'''

import numpy as np
import scipy as sp
from scipy import misc
from scipy import stats
import mpi4py.MPI as mpi
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import datetime as dt
import os
import sys

import Image
import cv2

import glob
import time

comm = mpi.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

cap = cv2.VideoCapture(sys.argv[1])
f_rate = cap.get(cv2.cv.CV_CAP_PROP_FPS)
frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
f_sec = 1 / f_rate
width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

#split files
rootfile, ext = sys.argv[1].split(".")

#CHECK IF FOLDER EXISTS
temp_folder = "temp_folder2/"
master_file = temp_folder + "all_averaged.jpg"

if rank == 0:
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

#h is the step size. Essentially you want one frame at a time!
h = 1

#just a random stating point
a = 0

#local_i is the number of frames each processor will use
local_i = frames / ( size )

#calculate interval that each process handles
local_startframe = a + ( rank ) * local_i * h
local_endframe = local_startframe + local_i * h - 1

#print local_startframe , local_endframe , frames

for i in xrange(local_startframe, local_endframe, int(f_rate)):
    #print i
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, i)
    success,image = cap.read()
    if i < frames:
      cv2.imwrite(temp_folder + "frame%.7d.jpg" % i, image)
    comm.Barrier()

'''
CREATE MASTER IMAGE

THIS PART WAS TAKEN FROM:
http://stackoverflow.com/questions/9251580/stacking-astronomy-images-with-python
'''

files = glob.glob(temp_folder + "/frame*.jpg")
files.sort()
del files[-1]
#print len(files)

print "I am about to start creating master image"
if rank == 0:
    image=Image.open(files[0])
    im=np.array(image,dtype=np.float32)
    for i in range(len(files)):
        #print i
        currentimage=Image.open(files[i])
        im += np.array(currentimage, dtype=np.float32)
    im /= len(files) # lowered brightness, with magic factor
    # clip, convert back to uint8:
    final_image = Image.fromarray(np.uint8(im.clip(0,255)))
    final_image.save(master_file, 'JPEG')
else:
    while not os.path.exists(master_file):
        print "I, rank %d, am going to sleep..." % rank
        time.sleep(1)

'''
Calculate chi squared on the images
'''
final_image = misc.imread(master_file)
frame_no = []
frame_chi = []

print "About to start doing Chi-squared"
for i_base in xrange(0, len(files), size):
#for i_base in range(len(files)):
    i = i_base + rank
    if i < len(files):
        print i
        images = misc.imread(files[i])
        temp_frame, extention = files[i].split(".")
        folder, temp_frame = temp_frame.split("frame")
        temp_frame = float(temp_frame) * f_sec
        #print i, temp_frame
        i_shape = images.shape
        x, y, z = i_shape
        ddof = x * y
        chi, pval = stats.mstats.chisquare(final_image,images)
        reduced = np.sum(chi) / ( ddof )
            
        frame_no.append(float(temp_frame))
        frame_chi.append(float(reduced))

#print "About to start zipping..."
with open(temp_folder + 'results.dat', 'a') as outfile:
    print "About to start zipping... I am rank ", rank
    zipped = zip(frame_no, frame_chi)
    np.savetxt(outfile, zipped)
comm.Barrier()
    #THIS FILE OUTPUT WILL BE USED FOR POSTPROCESSING IF NEEDS BE


if rank == 0:
    print "I am about to start drawing those amazing figures you really want to see!"
    majorLocator   = MultipleLocator(5)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator   = MultipleLocator(1)

    outfile = temp_folder + 'results.dat'
    frame_no, frame_chi = np.loadtxt(outfile, usecols=(0,1), unpack=True)
    zipped = zip(frame_no, frame_chi)
    zipped.sort()
    np.savetxt(outfile, zipped)
    frame_no, frame_chi = np.loadtxt(outfile, usecols=(0,1), unpack=True)

    #frame_no, frame_chi = zip(zipped)
    #print frame_chi.min

    frameTime = frame_no / 60
    redChi = frame_chi
    
    minTime, maxTime = min(frameTime), max(frameTime)
    minChi, maxChi = min(redChi), max(redChi)
    averageChi = float(sum(redChi) / len(redChi))

    plt.axhline(y=3*minChi)

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
    plt.text(xpos, ypos, rootfile)

    plt.plot(frameTime,redChi, 'r-')
    #plt.savefig(temp_folder + "reduced.eps")
    #plt.savefig(rootfile + ".eps")
    plt.savefig(rootfile + ".jpg")

    for i in range(len(frameTime)):
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
TO CREATE MOVIE
http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
'''

# Define the codec and create VideoWriter object
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter(rootfile + '.avi',fourcc, 1, (width,height))

if rank == 0:
    print "Making a movie"
    x_offset=y_offset= 0
    files = glob.glob(temp_folder + "/frame*.jpg")
    files.sort()
    del files[-1]
    output = glob.glob(temp_folder + "image*.jpg")
    output.sort()
    for i in xrange(0, len(files), 1):
        if frame_chi[i] > 3*minChi:
            images = cv2.imread(files[i])
            s_img = cv2.imread(output[i])
            s_img = cv2.resize(s_img, (0,0), fx=0.5, fy=0.5) 
            images[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
            out.write(images)

cap.release()
cv2.destroyAllWindows()
out.release()

