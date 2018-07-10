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
    PARALLAX = 64
    PATCH_SIZE = 33
    STEP = PARALLAX
    STRIDE = 24
    BATCH_SIZE = 128
    DATA_AUG_TIMES = 2
    SRCDIR = '../Data/mb2014_png/'

    fpdata_tr  = sorted(glob.glob(SRCDIR + 'train/X_left/*.png'))
    numPics_tr = len(fpdata_tr)

    DSTDIR = './data/'
    FDATA = DSTDIR + ('data_da%d_p%d_s%d_b%d_par%d_tr%d.hdf5' 
                    % (DATA_AUG_TIMES, PATCH_SIZE, STRIDE, BATCH_SIZE, PARALLAX, numPics_tr))
    SAVEPROB = 1
    CHKDIR = './data/chk'

    if not os.path.exists(DSTDIR):
        os.makedirs(DSTDIR)
    if not os.path.exists(CHKDIR):
        os.makedirs(CHKDIR)
    subprocess.check_call('rm -f {}/*'.format(CHKDIR), shell=True)
    
    count = 0
    for i in xrange(numPics_tr):
        fleft  = SRCDIR + ('train/X_left/X%d.png') % (i+1)
        img = Image.open(fleft)
        img_rs = img.resize((int(img.size[0]/2), int(img.size[1]/2)))
        im_h, im_w = img_rs.size
        for x in range(0+STEP, (im_h-PATCH_SIZE), STRIDE):
            for y in range(0+STEP, (im_w-PATCH_SIZE), STRIDE):
                count += 1
    origin_patch_num = count * DATA_AUG_TIMES
    if origin_patch_num % BATCH_SIZE != 0:
        numPatches = (origin_patch_num/BATCH_SIZE+1) * BATCH_SIZE
    else:
        numPatches = origin_patch_num

    print("[*] Info ..")
    print("\t Number of train images = %d" % numPics_tr)
    print("\t Number of patches = %d" % numPatches)
    print("\t Patch size = %d" % PATCH_SIZE)
    print("\t Batch size = %d" % BATCH_SIZE)
    print("\t Number of batches = %d" % (numPatches/BATCH_SIZE))
    print("\t DATA_AUG_TIMES = %d" % DATA_AUG_TIMES)
    print("\t Source dir = %s" % SRCDIR)
    print("\t Dest dir = %s" % DSTDIR)
    print("\t Dest file = %s" % FDATA)
    sys.stdout.flush()

    shape_tr_in   = (numPatches, PATCH_SIZE, PATCH_SIZE, 3 * (PARALLAX+1))
    shape_tr_out  = (numPatches, PATCH_SIZE, PATCH_SIZE, 3)

    hdfile = h5py.File(FDATA, mode = 'w')
    hdfile.create_dataset("X_tr" , shape_tr_in  , np.uint8)
    hdfile.create_dataset("Y_tr" , shape_tr_out , np.uint8)

    print("[*] Processing Train Images")
    
    c = 0
    for i in xrange(numPics_tr):
        print("\t Tr image [%2d/%2d]" % (i+1, numPics_tr))
        sys.stdout.flush()

        fleft  = SRCDIR + 'train/X_left/X%d.png' % (i+1)
        fright = SRCDIR + 'train/X_right/X%d.png' % (i+1)
        fyleft = SRCDIR + 'train/Y_left/Y%d.png' % (i+1)
        imgL  = Image.open(fleft)
        imgR  = Image.open(fright)
        imgyL = Image.open(fyleft)

        imgL_rs  = imgL.resize( (int(imgL.size[0]/2) , int(imgL.size[1]/2)) , resample=PIL.Image.BICUBIC)
        imgR_rs  = imgR.resize( (int(imgR.size[0]/2) , int(imgR.size[1]/2)) , resample=PIL.Image.BICUBIC)
        imgyL_rs = imgyL.resize((int(imgyL.size[0]/2), int(imgyL.size[1]/2)), resample=PIL.Image.BICUBIC)

        h, w = imgL_rs.size
        imgL_np  = np.reshape(np.array(imgL_rs , dtype='uint8'), (w,h,3))
        imgR_np  = np.reshape(np.array(imgR_rs , dtype='uint8'), (w,h,3))
        imgyL_np = np.reshape(np.array(imgyL_rs, dtype='uint8'), (w,h,3))

        im_h, im_w, _ = imgL_np.shape
        for j in xrange(DATA_AUG_TIMES):
            for x in range(0+STEP, im_h-PATCH_SIZE, STRIDE):
                for y in range(0+STEP, im_w-PATCH_SIZE, STRIDE):
                    mode = random.randint(0, 7)
                    xx = np.zeros((1,PATCH_SIZE, PATCH_SIZE, 3*(PARALLAX+1)))
                    xx[0,:,:,0:3] = data_augmentation(imgL_np[x:x+PATCH_SIZE,y:y+PATCH_SIZE,:], mode)
                    pp = 0
                    for p in range(3,3*(PARALLAX+1), 3):
                        xx[0,:,:,p:p+3] = data_augmentation(imgR_np[x:x+PATCH_SIZE,y-pp:y+PATCH_SIZE-pp,:], mode)
                        pp += 1
                    y = data_augmentation(imgyL_np[x:x+PATCH_SIZE,y:y+PATCH_SIZE,:], mode)
                    hdfile["X_tr"][c, ...] = xx
                    hdfile["Y_tr"][c, ...] = y
                    if random.random() > SAVEPROB:
                        for p in range(0,3*(PARALLAX+1),3):
                            x_im = xx[0,:,:,p:p+3].astype("uint8")
                            Image.fromarray(x_im).save(CHKDIR + ('/%d_x_%d.png' % (c,p)),'png')

                        y_im = y.astype("uint8")
                        Image.fromarray(y_im).save(CHKDIR + ('/%d_y.png' % (c)),'png')
                    c += 1

if __name__ == '__main__':
    generate_hdf5()
