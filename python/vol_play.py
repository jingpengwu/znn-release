# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 17:11:42 2014

@author: jingpeng
"""

import numpy as np
import emirt
import os

# parameters
batchid = 1
# affinity direction, 0,1,2
ad = 1
chann_fname = '../dataset/fish/data/batch{}.image'.format( batchid )
out_fname = '../experiments/VeryDeep2HR_w65x9/output/out{}.{}'.format( batchid, ad )
#out_fname = '../experiments/VeryDeep2_w109/output/out{}.1'.format( batchid )
# out_fname = '../experiments/W3_C10_P2_D2/output/out{}.1'.format( batchid )
# out_fname = '../experiments/VGG_L8_3R/output.a/out{}.2'.format( batchid )
#out_fname = '../experiments/W3_C10_P2_D2/output/out{}.1'.format( batchid )
#out_fname = '../experiments/W4_C13_P3_D3/output/out{}.1'.format( batchid )
#out_fname = '../experiments/N4/output2/out{}.1'.format( batchid )
#out_fname = '../experiments/W59_C10_P3_D3/output/out{}.2'.format( batchid )
#out_fname = '../experiments/W5_C10_P3_D2/output/out{}.1'.format( batchid )
#out_fname = '../experiments/W9_C8_P3_D3/output/out{}.2'.format( batchid )
#out_fname = '../experiments/W10_C10_P3_D3/output/out{}.2'.format( batchid )

# AWS filenames
out_fname = ("jingpeng1", "/home/znn-release/", "experiments/W10_C10_P3_D3/output/", "out{}.{}".format( batchid, ad ))
#out_fname = ("jingpeng2", "/home/znn-release/", "experiments/VeryDeep2HR_w65x9/output/", "out{}.{}".format( batchid, ad ))
#out_fname = ("jingpeng3", "/home/znn-release/", "experiments/W14_C8_P3_D3/output/", "out{}.{}".format( batchid, ad ))

#%% download the data from AWS
if isinstance(out_fname, tuple):
    # download the train and test info
    os.system("starcluster get "+ out_fname[0] + " " + out_fname[1]+
                    out_fname[2]+ out_fname[3] + "* ../"+ out_fname[2])
    out_fname = "../"+ out_fname[2]+out_fname[3]
    
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
out1 = emirt.io.znn_img_read( out_fname )
zo,yo,xo = out1.shape
# read channel data
chann = emirt.io.znn_img_read( chann_fname )
zc,yc,xc = chann.shape

# align the chann and out
out = np.zeros(chann.shape)
zoff = (zc - zo)/2
yoff = (yc - yo)/2
xoff = (xc - xo)/2
out[ zoff:zc-zoff, yoff:yc-yoff, xoff:xc-xoff ] = out1


#%%
import emirt
#emirt.show.vol_slider(vol, cmap='gray')
com = emirt.show.CompareVol(chann, out)
com.vol_compare_slice()
