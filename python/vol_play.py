# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 17:11:42 2014

@author: jingpeng
"""

import numpy as np

# parameters
chann_fname = '../dataset/fish/data/batch92.image'
out_fname = '../experiments/VeryDeep2_w109/output/out92.1'
#out_fname = '../experiments/VGG_L9/output/out92.1'
#out_fname = '../experiments/Deep_N4/output/out92.1'

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
out = np.fromfile(out_fname, dtype='double')
out_sz = np.fromfile(out_sz_fname, dtype='uint32')[:3][::-1]
out = out.reshape(out_sz)

# read channel data
chann = np.fromfile(chann_fname, dtype='double')
chann_sz = np.fromfile( chann_sz_fname, dtype='uint32' )[:3][::-1]
chann = chann.reshape(chann_sz)


#%%
import emirt
#emirt.show.vol_slider(vol, cmap='gray')
com = emirt.show.CompareVol(chann[:,:,:], out)
com.vol_compare_slice()
import time
time.sleep(60)
