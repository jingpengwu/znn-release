# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:52:04 2015

@author: jingpeng
"""
import emirt
import numpy as np
import tifffile
#%% parameters
srcImg = '/usr/people/jingpeng/seungmount/Omni/TracerTasks/ZfishAnnotation/Merlin/Export_Merlin_Daan-corrected/Merlin_label.tif'
dstImg = 'test.tif'

#%% read volume
vol = emirt.io.imread( srcImg )

#%% transform
bw = np.sum(vol, axis=3, dtype='uint8')
bw[bw>0] = 255

#%% write the image volume
from tifffile import imsave
imsave(dstImg, bw[102:,:,:])