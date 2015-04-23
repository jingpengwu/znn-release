# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 09:47:47 2015

label variation for training

@author: jingpeng
"""
import tifffile
#%% parameters
srcLbl = '../dataset/fish/data/original/Daan_label.tif'
dstLbl = '../dataset/fish/data/original/Daan_label_be.tif'

srcLbl = '../dataset/fish/data/original/Kyle_label.tif'
dstLbl = '../dataset/fish/data/original/Kyle_label_be.tif'

srcLbl = '../dataset/fish/data/original/Merlin_label1.tif'
dstLbl = '../dataset/fish/data/original/Merlin_label1_be.tif'

#%% read volume
lbl = tifffile.imread(srcLbl)

#%% morphological operation
import scipy.ndimage
structure = scipy.ndimage.generate_binary_structure(3, 1)
structure[0,:,:] = False
structure[2,:,:] = False
lbl2 = scipy.ndimage.binary_erosion(lbl, structure=structure).astype( lbl.dtype )*255
# border voxel should be the same with original label
lbl2[:,0, :] = lbl[:,0, :]
lbl2[:,-1,:] = lbl[:,-1,:]
lbl2[:,:, 0] = lbl[:,:, 0]
lbl2[:,:,-1] = lbl[:,:,-1]

#%% save the volume
tifffile.imsave(dstLbl, lbl2)

#%% compare two volumes
import emirt
reload(emirt)
com = emirt.show.compare_vol(lbl, lbl2)
com.vol_compare_slice()