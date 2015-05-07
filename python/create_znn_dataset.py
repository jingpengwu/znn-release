# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:11:45 2015

@author: jingpeng
"""
import numpy as np

#%% parameters
Dir = '../dataset/fish/'
fnm_trn = Dir + 'data/original/Merlin_raw1.tif'
fnm_lbl = Dir + 'data/original/Merlin_label1.tif'

# the batch id
batch_id = 3

#%% read volume
import emirt.io
vlm_trn = emirt.io.imread( fnm_trn ).transpose(0,2,1)
vlm_lbl = emirt.io.imread( fnm_lbl ).transpose(0,2,1)
sz = np.asarray( vlm_trn.shape )
#%% save as znn format
emirt.io.znn_img_save(vlm_trn.astype('double'), Dir+'data/batch{}'.format(batch_id)+".image")
emirt.io.znn_img_save(vlm_lbl.astype('double'), Dir+'data/batch{}'.format(batch_id)+".label")

#%% prepare corresponding spec file
f = open( Dir + 'spec/batch{}'.format(batch_id) + ".spec", 'w')
f.write( """[INPUT1]
path=./dataset/fish/data/batch{0}
ext=image
size={1},{2},{3}
pptype=standard2D

[LABEL1]
path=./dataset/fish/data/batch{0}
ext=label
size={1},{2},{3}
pptype=binary_class

[MASK1]
size={1},{2},{3}
pptype=one
ppargs=2""".format(batch_id, sz[2], sz[1], sz[0]) )
f.close()

#%%
#import emirt.show
#cmp = emirt.show.CompareVol(vlm_lbl, vlm_trn)
#cmp.vol_compare_slice()

#%% compare two binary files
vm = emirt.io.znn_img_read(Dir+'data/00_backup/batch1.label')
sz = np.fromfile(Dir+'data/batch1.size', dtype='uint32')
vp = emirt.io.znn_img_read(Dir+'data/batch1.label')
print "matlab==python? {}".format( np.all( vm==vp ) )

szm = np.fromfile(Dir+'data/00_backup/batch1.size', dtype='uint32')
szp = np.fromfile(Dir+'data/batch1.size', dtype='uint32')
print "matlab==python? {}".format( np.all( szm==szp ) )