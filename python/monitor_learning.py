# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 09:52:48 2014

@author: jingpeng
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import savitzky_golay as sg


# parameters
# directories
Dir = '../experiments/VeryDeep2_w109_V1/network/'
#Dir = '../experiments/VGG_L10/network/'
#Dir = '../experiments/Deep_N4/network/'
Dir = '../experiments/VeryDeep2HR_w65x9/network/'

# parameters to smooth curve
smooth_flag = True
Window_size = 25    # must be a odd number
Polynomial_order = 3


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
    train_err = sg.savitzky_golay(train_err, Window_size, Polynomial_order)
    train_cls = sg.savitzky_golay(train_cls, Window_size, Polynomial_order)
    test_err = sg.savitzky_golay(test_err, Window_size, Polynomial_order)
    test_cls = sg.savitzky_golay(test_cls, Window_size, Polynomial_order)
     
# adjust the range
train_minEle= min(len(train_err), len(train_cls), len(train_iter))
train_iter = train_iter[:train_minEle]
train_err = train_err[:train_minEle]
train_cls = train_cls[:train_minEle]

test_minEle = min(len(test_err), len(test_cls), len(test_iter))
test_iter = test_iter[:test_minEle]
test_err  = test_err[:test_minEle]
test_cls  = test_cls[:test_minEle]

# plot 
fig = plt.figure(figsize=(15,5))

fig_err = fig.add_subplot(121)
fig_err.plot(train_iter, train_err,'-k', test_iter, test_err, '-r')
fig_err.set_title('cost')
fig_err.set_ylabel('cross entropy')
fig_err.set_xlabel('iterartion')
fig_err.set_ylim([0, train_err.max()])
fig_err.legend(('train', 'test'), loc=0)

fig_cls = fig.add_subplot(122)
fig_cls.plot(train_iter, train_cls,'-k', test_iter, test_cls, '-r')
fig_cls.set_title('classification error')
fig_cls.set_ylabel('classification error')
fig_cls.set_xlabel('iteration')
fig_cls.set_ylim([0, train_cls.max()])
fig_cls.legend(('train', 'test'), loc=0)
fig.show()
