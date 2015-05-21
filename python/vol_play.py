# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 17:11:42 2014

@author: jingpeng
"""

import numpy as np

# parameters
batchid = 91
#chann_fname = '../dataset/fish/data/batch{}.image'.format( batchid )
out_fname = '../experiments/VeryDeep2HR_w65x9/output/out{}.2/usr/people/jingpeng/seungmount/research/Jingpeng/01_ZNN/znn-release/networks/VeryDeep2_w109.spec'.format( batchid )
#out_fname = '../experiments/VeryDeep2_w109/output/out{}.1'.format( batchid )
out_fname = '../experiments/VGG_L7/output/out{}.1'.format( batchid )
#out_fname = '../experiments/Deep_N4/output/out{}.1'.format( batchid )

#%% process file name
def get_size_fname( fname ):
    if '.image' in fname:
        sfname = fname.replace('.image', '.size')
    elif '.label' in fname:
        sfname = fname.replace('.label', '.size')
    else:
        sfname = fname + '.size'
    return sfname

out_sz_fname   = get_size_fname(out_fname)
chann_sz_fname = get_size_fname(chann_fname)

#%% readvolume
# read output data
out = emirt.io.znn_img_read( out_fname )

# read channel data
chann = emirt.io.znn_img_read( chann_fname )
chann = chann[0:,54:,54:]

#%%
import emirt
#emirt.show.vol_slider(vol, cmap='gray')
com = emirt.show.CompareVol(chann[:,:,:], out)
com.vol_compare_slice()
#import time
#time.sleep(60)
