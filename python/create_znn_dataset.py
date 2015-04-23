# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:11:45 2015

@author: jingpeng
"""
import numpy as np
import emirt
#%% parameters
Dir = '../dataset/fish/'
fnm_trn = Dir + 'data/original/Daan_raw.tif'
fnm_lbl = Dir + 'data/original/Daan_label.tif'

# the batch id
batch_id = 71

#%% read volume
vlm_trn = emirt.io.imread( fnm_trn )
vlm_lbl = emirt.io.imread( fnm_lbl )
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