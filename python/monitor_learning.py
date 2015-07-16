#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 09:52:48 2014

@author: jingpeng
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os

#%% parameters
# directories
Dir = '../experiments/VeryDeep2_w109/network_overfitting/'
#Dir = '../experiments/VGG_L7_3/network/'
#Dir = '../experiments/VGG_L10/network/'
# Dir = '../experiments/VGG_L8_3R/network.a/'
#Dir = '../experiments/N4/network/'
Dir = '../experiments/VeryDeep2HR_w65x9_bak/network/'
# Dir = '../experiments/VeryDeep2HR_w65x9_malis/network/'
#Dir = '../experiments/VeryDeep2_S1/network/'
#Dir = '../experiments/N4/network/'
#Dir = '../experiments/N4_2R/network/'
#Dir = '../experiments/W2_L8_3/network/'
#Dir = '../experiments/W3_C10_P2_D2/network/'
# Dir = '../experiments/W4_C13_P3_D3/network/'
Dir = '../experiments/W5_C10_P3_D2/network2/'
#Dir = '../experiments/W6_C13_P3_D3/network/'
#Dir = '../experiments/W7_C11_P3_D3/network/'
#Dir = '../experiments/W9_C8_P3_D3/network/'
#Dir = '../experiments/W59_C10_P3_D3/network/'
#Dir = '../experiments/W12_C5_P3_D3/network/'
#Dir = '../experiments/W10_C10_P3_D3/network/'
#Dir = '../experiments/W14_C9_P3_D3_A/network/'

# AWS training, fist: cluster name, second: path
#Dir = ("jingpeng1", "/home/znn-release/experiments/W10_C10_P3_D3/network/")
#Dir = ("jingpeng2", "/home/znn-release/experiments/VeryDeep2HR_w65x9/network/")
#Dir = ("jingpeng3", "/home/znn-release/experiments/W14_C8_P3_D3/network/")

# parameters to smooth curve
smooth_flag = True
Window_size = 25    # must be a odd number
Polynomial_order = 3

#%% process directory of AWS training
if isinstance(Dir, tuple):
    # download the train and test info
    os.system("starcluster get "+ Dir[0] + " "+Dir[1]+"t* /tmp/")
    Dir = "/tmp/"


#%% smooth function
def savitzky_golay(y, window_size, order, deriv=0, rate=1):

    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

# read train and test data
train_iter = np.fromfile(Dir + 'train.iter', dtype='uint64')
train_err = np.fromfile(Dir + 'train.err', dtype='double')
train_cls = np.fromfile(Dir + 'train.cls', dtype='double')

test_iter = np.fromfile(Dir + 'test.iter', dtype='uint64')
test_err = np.fromfile(Dir + 'test.err', dtype='double')
test_cls = np.fromfile(Dir + 'test.cls', dtype='double')


# output training and testing results
print 'number of iterations:                    {0}     '.format(train_iter[-1])
print 'the minimum:'
print 'cost of train and test:                  {0}   {1}'.format(min(train_err[-100:]), min(test_err[-100:]))
print 'classification error of train and test:  {0}   {1}'.format(min(train_cls[-100:]), min(test_cls[-100:]))

print 'the median:'
print 'cost of train and test:                  {0}   {1}'.format(sp.median(train_err[-100:]), sp.median(test_err[-100:]))
print 'classification error of train and test:  {0}   {1}'.format(sp.median(train_cls[-100:]), sp.median(test_cls[-100:]))

# smooth curve using Savitzky-Golay filter
if smooth_flag:
    train_err = savitzky_golay(train_err, Window_size, Polynomial_order)
    train_cls = savitzky_golay(train_cls, Window_size, Polynomial_order)
    test_err = savitzky_golay(test_err, Window_size, Polynomial_order)
    test_cls = savitzky_golay(test_cls, Window_size, Polynomial_order)

# adjust the range
train_minEle= min(len(train_err), len(train_cls), len(train_iter))
train_iter = train_iter[:train_minEle]
train_err = train_err[:train_minEle]
train_cls = train_cls[:train_minEle]

test_minEle = min(len(test_err), len(test_cls), len(test_iter))
test_iter = test_iter[:test_minEle]
test_err  = test_err[:test_minEle]
test_cls  = test_cls[:test_minEle]

#%% plot
fig = plt.figure(figsize=(15,5))

fig_err = fig.add_subplot(121)
fig_err.plot(train_iter, train_err,'-k', test_iter, test_err, '-r')
fig_err.set_title('cost')
fig_err.set_ylabel('cross entropy')
fig_err.set_xlabel('iterartion')
#fig_err.set_ylim([0, max(train_err.max(), train_cls.max())])
fig_err.legend(('train', 'test'), loc=0)

fig_cls = fig.add_subplot(122)
fig_cls.plot(train_iter, train_cls,'-k', test_iter, test_cls, '-r')
fig_cls.set_title('classification error')
fig_cls.set_ylabel('classification error')
fig_cls.set_xlabel('iteration')
#fig_cls.set_ylim([0, max(train_err.max(), train_cls.max())])
fig_cls.legend(('train', 'test'), loc=0)
fig.show()
plt.show()
