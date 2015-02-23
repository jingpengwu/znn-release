# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 17:11:42 2014

@author: jingpeng
"""

import numpy as np

# parameters
#Dir = '/usr/people/jingpeng/experiments/'
#Dir = '/usr/people/jingpeng/seunglab4/experiments/'
Dir = '/usr/people/jingpeng/znn-release-master/experiments/Deep_N4/output/'
#Dir = '/usr/people/jingpeng/seungmount/research/Jingpeng/01_workspace/01_ZNN/21_experiments/P09_N3_experiments/'
#Dir = '/usr/people/jingpeng/seungmount/research/Jingpeng/01_workspace/01_ZNN/21_experiments/P10_N4_experiments/'
#Dir = '/usr/people/jingpeng/seungmount/research/kisuklee/znn-release/experiments/Ashwin/3D_affinity/train23_test1/VeryDeep2HR_w65x9/rebalanced/eta02_out100/iter_60K/output/'
#Dir = '/usr/people/jingpeng/seungmount/research/Jingpeng/01_workspace/03_watershed/01_data/'
#Dir = '/usr/people/jingpeng/seungmount/research/kisuklee/znn-release/experiments/Ashwin/3D_affinity/train23_test1/VeryDeep2H_w65x9/unbalanced/eta02_out100/iter_60K/output/'
Dir = '/usr/people/jingpeng/seungmount/research/Jingpeng/01_workspace/06_ZNN_SSE/znn-release-master/experiments/N4_V1/output/'

#%% reading the volume
fname = Dir + "out1.1"
if '1.1' in fname:
    fp_label = r"/usr/people/jingpeng/znn-release-master/dataset/fish/data/original/ExportBoundaries_8bit_Daan.tif"
    fp_origin = r"/usr/people/jingpeng/znn-release-master/dataset/fish/data/original/RawInput_8bit_Daan_train.tif"
elif '2.1' in fname:
    fp_origin = r"/usr/people/jingpeng/znn-release-master/dataset/fish/data/original/RawInput_8bit_Daan_test.tif"
    fp_label = []
elif '4.1' in fname:
    fp_origin = []
    fp_label = []
else:
    print 'illigal file name! Please check the file name!'
    
#fname = "/usr/people/jingpeng/znn-release-master/experiments/out1.0"
vol = np.fromfile(fname, dtype='double')
sz = np.fromfile(fname+'.size', dtype='uint32')
vol_result = vol.reshape(sz, order='F')
#vol_result = vol_result.transpose()

import tifffile
if fp_label:
    imfile = tifffile.TIFFfile(fp_label)
    vol_label = imfile.asarray().transpose((1,2,0))
else:
    vol_label=[]

imfile = tifffile.TIFFfile(fp_origin)
vol_origin = imfile.asarray().transpose((1,2,0))
#np.transpose

#%% display image
HL_sec = 1 #highlight,candidates: 46
# offset
xyoff = 32
zoff = 1

Nx, Ny, Nz = vol_result.shape
# get the images
im_origin = vol_origin[xyoff:xyoff+Nx,  xyoff:xyoff+Ny, HL_sec+zoff]
im_result = uint8(vol_result[:,:,HL_sec]*255)

if fp_label:
    im_label  = vol_label[xyoff:xyoff+Nx,   xyoff:xyoff+Ny, HL_sec+zoff]

# write the images
#import jptools
#jptools.imwrite(im_origin, 'test_sec'+str(HL_sec) + '_origin.tif')
#jptools.imwrite(im_result, 'test_sec'+str(HL_sec) + '_result.tif')
#if fp_label:
#    jptools.imwrite(im_label, 'test_sec'+str(HL_sec) + '_label.tif')
#vol_origin_crop = vol_origin[xyoff:xyoff+Nx,  xyoff:xyoff+Ny, zoff:zoff+Nz]
#vol_origin_crop.tofile(Dir +'train1.1_crop')

#vol_origin_crop.tofile(Dir +'bach_crop1.label')

vol_origin_label = vol_label[xyoff:xyoff+Nx,  xyoff:xyoff+Ny, zoff:zoff+Nz]

#%%
import matplotlib.pyplot as plt
#fig = plt.figure(figsize=(30,6))
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
ax.imshow(im_origin, cmap = plt.gray())
ax.set_title('original image')
ax.set_xticks([])
ax.set_yticks([])

if fp_label:
    ax2 = fig.add_subplot(1, 3, 2, sharex=ax, sharey=ax)
    ax2.imshow(im_label, cmap = plt.gray())
    ax2.set_title('labeled image')
    ax2.set_xticks([])
    ax2.set_yticks([])

ax3 = fig.add_subplot(1, 3, 3, sharex=ax, sharey=ax)
ax3.imshow(im_result, cmap = plt.gray())
ax3.set_title('ZNN result')
ax3.set_xticks([])
ax3.set_yticks([])
fig.show()

# total volume
#vol = np.concatenate((vol_label, vol_result), axis=1)

#%% 3D display
import pyqtgraph as pg
imv = pg.ImageView()
imv.show()
imv.setImage(vol_result.transpose())

imv_train = pg.ImageView()
imv_train.show()
imv_train.setImage(vol_origin.transpose())

if fp_label:
    imv_label = pg.ImageView()
    imv_label.show()
    imv_label.setImage(vol_label.transpose())
    

# 3D volume rendering using mayavi
#from mayavi import mlab
#mlab.pipeline.volume(mlab.pipeline.scalar_field(vol_origin))
#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(vol_origin),
#                            plane_orientation='z_axes',
#                            slice_index=10,
#                        )