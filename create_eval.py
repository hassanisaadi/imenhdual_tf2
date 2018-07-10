#! /usr/bin/env python2

import glob
import numpy as np
import h5py
import sys
from PIL import Image
import PIL
import subprocess
import random
import os
from utils import *

def generate_hdf5():
    SRCDIR = '../Data/mb2014_png/eval/'

    fpdata_eval   = sorted(glob.glob(SRCDIR + 'X_left/*.png'))
    numPics_eval   = len(fpdata_eval)

    DSTDIR = './data/eval/'

    if not os.path.exists(DSTDIR):
        os.makedirs(DSTDIR)
    
    c = 0
    for i in xrange(numPics_eval):
        print("\t eval image [%2d/%2d]" % (i+1, numPics_eval))
        sys.stdout.flush()

        fleft  = SRCDIR + 'X_left/XL%d.png' % (i+191)
        fright = SRCDIR + 'X_right/XR%d.png' % (i+191)
        fyleft = SRCDIR + 'Y_left/YL%d.png' % (i+191)
        imgL  = Image.open(fleft)
        imgR  = Image.open(fright)
        imgyL = Image.open(fyleft)

        imgL_rs  = imgL.resize( (int(imgL.size[0]/2) , int(imgL.size[1]/2)) , resample=PIL.Image.BICUBIC)
        imgR_rs  = imgR.resize( (int(imgR.size[0]/2) , int(imgR.size[1]/2)) , resample=PIL.Image.BICUBIC)
        imgyL_rs = imgyL.resize((int(imgyL.size[0]/2), int(imgyL.size[1]/2)), resample=PIL.Image.BICUBIC)

        imgL_rs.save(DSTDIR + 'XL%d.png' % i)
        imgR_rs.save(DSTDIR + 'XR%d.png' % i)
        imgyL_rs.save(DSTDIR + 'YL%d.png' % i)

if __name__ == '__main__':
    generate_hdf5()
