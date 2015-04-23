# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 18:20:28 2015

@author: jingpeng
"""
import numpy as np
import h5py
import tifffile

#%% read hdf5 volume
def imread( fname ):
    if '.hdf5' in fname:
        fname = fname.replace(".hdf5", "")
        f = h5py.File( fname + '.hdf5' )
        v = np.asarray( f['/main'] )
        f.close()
        print 'finished reading image stack :)'
        return v
    elif '.tif' in fname:
        return tifffile.imread( fname )
    else:
        print 'file name error, only suport tif and hdf5 now!!!'
        

def imwrite( vol, fname ):
    if '.hdf5' in fname:
        f = h5py.File( fname )
        f.create_dataset('/main', data=vol)
        f.close()
        print 'hdf5 file was written :)'
    elif '.tif' in fname:
        tifffile.imsave(fname, vol)
    else:
        print 'file name error! only support tif and hdf5 now!!!'

def bar_plot(vector, label):
    import matplotlib.pylab as plt
    plt.bar(vector, label)

#%% get the supervoxel list of each segment of user u
def get_list_supervoxels( vu, vr ):
    # number of supervoxels of user's labling
    lst_segid_user = np.unique(vu)[1:]
    
    lst_supervoxels_in_segments_in_user = list( lst_segid_user )
    
    # get the supervoxel list
    for idx, segid in enumerate(lst_segid_user):
        lst_supervoxels_in_segments_in_user[idx] = np.unique( vr[ vu == segid ] )
        
    return lst_supervoxels_in_segments_in_user

#%% get the list of supervoxels in segments of each user
def get_lst_supervoxels_in_segments_in_users(Dir, users):
    print 'get supervoxel list of each segment of every user'
    vr = imread( Dir + 'Raw.hdf5' )
    # initialization
    lst_supervoxels_in_segments_in_users = list( range( len(users) ) )
    # get the list
    for uidx, user in enumerate(users):
        vu = imread(Dir + user + '.tif')
        lst_supervoxels_in_segments_in_users[uidx] = get_list_supervoxels( vu, vr )
        print 'completed user ID: {}'.format( uidx )
        
    print 'finish getting supervoxel list of every user'
    # return finall result
    return lst_supervoxels_in_segments_in_users

#%% see 3D consensus
def show_3d_slices(vol):
    # transform to random color
    
    # transpose for correct show
    vol = vol.transpose()
    
    # show the image stack
    import pyqtgraph as pg
#    cm = np.random.rand(np.max(vol))
#    pg.colormap(cm)
    imv = pg.ImageView()
    imv.show()
    imv.setImage(vol)
    
# 3D volume rendering using mayavi
def mayavi_3d_rendering(vol):
    vol = vol.transpose()
    from mayavi import mlab
#    mlab.pipeline.volume(mlab.pipeline.scalar_field(vol))
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(vol),
                                plane_orientation='z_axes',
                                slice_index=10,
                                )