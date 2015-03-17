# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:19:15 2015

@author: jingpeng
"""
import numpy as np
import matplotlib.pylab as plt

#%% convolution with different kernel size, source image size=200X200X1
# kernel size
ksize = np.array([3,6,12,24,48,96])

# naive and MKL method user time
ntime = np.array([1.00/100, 4.14/100, 1.42/10, 4.90/10, 14.7/10, 28.2/10])
dtime = np.array([0.26/100, 0.66/100, 0.22/10, 0.74/10, 2.12/10, 3.90/10]) # DIRECT
ftime = np.array([3.29/100, 3.93/100, 0.45/10, 0.37/10, 0.52/10, 0.82/10]) # FFT
#atime = np.array([3.82/100, 3.72/100, 5.23/100, 0.40/10, 0.52/10, 0.77/10]) # AUTO

# speed up
sud = ntime / dtime
suf = ntime / ftime
#sua = ntime / atime

# plot
plt.subplot(121)
plt.plot(ksize, ntime, '-ro', label='NAIVE')
plt.plot(ksize, dtime, '-gp', label='MKL--DIRECT')
plt.plot(ksize, ftime, '-b*', label='MKL--FFT')
plt.xlabel('kernel size')
plt.ylabel('user time (s)')
plt.legend(loc=0)

plt.subplot(122)
plt.plot(ksize, sud, '-gp', label='DIRECT')
plt.plot(ksize, suf, '-b*', label='FFT')
plt.legend(loc=0)
plt.xlabel('kernel size')
plt.ylabel('speed up (X)')
#plt.title('convolution with different kernel size')

#%% convolution with different source image size, kernel size = 6X6X1
# source image size
ssize = np.array([12,24,48,96,192])

# naive and MKL method user time
ntime = np.array([0.51/10000, 3.43/10000, 1.87/1000, 8.79/1000, 3.55/100])
dtime = np.array([0.12/10000, 0.78/10000, 0.30/1000, 1.20/1000, 0.58/100])
ftime = np.array([2.01/10000, 5.15/10000, 1.61/1000, 9.15/1000, 3.01/100])

# speed up
sud = ntime / dtime
suf = ntime / ftime

# plot
plt.subplot(121)
plt.plot(ssize, ntime, '-ro', label='NAIVE')
plt.plot(ssize, dtime, '-gp', label='MKL--DIRECT')
plt.plot(ssize, ftime, '-b*', label='MKL--FFT')
plt.xlabel('image size')
plt.ylabel('user time (s)')
plt.legend(loc=0)

plt.subplot(122)
plt.plot(ssize, sud, '-gp', label='DIRECT')
plt.plot(ssize, suf, '-b*', label='FFT')
plt.legend(loc=0)
plt.xlabel('image size')
plt.ylabel('speed up (X)')

#%% sparse convolution with different kernel size, source image size=200X200X1
# kernel size
ksize = np.array([3,6,12,24,48,96])

# naive and MKL method user time
ntime = np.array([0.54/100, 1.12/100, 3.90/100, 1.37/10, 4.12/10, 7.77/10])
dtime = np.array([0.24/100, 0.32/100, 0.75/100, 0.19/10, 0.55/10, 1.02/10]) # DIRECT
ftime = np.array([3.74/100, 3.78/100, 5.19/100, 0.40/10, 0.50/10, 0.78/10]) # FFT
#atime = np.array([3.82/100, 3.72/100, 5.23/100, 0.40/10, 0.52/10, 0.77/10]) # AUTO

# speed up
sud = ntime / dtime
suf = ntime / ftime
#sua = ntime / atime

# plot
plt.subplot(121)
plt.plot(ksize, ntime, '-ro', label='NAIVE')
plt.plot(ksize, dtime, '-gp', label='MKL--DIRECT')
plt.plot(ksize, ftime, '-b*', label='MKL--FFT')
plt.xlabel('kernel size')
plt.ylabel('user time (s)')
plt.legend(loc=0)

plt.subplot(122)
plt.plot(ksize, sud, '-gp', label='DIRECT')
plt.plot(ksize, suf, '-b*', label='FFT')
plt.legend(loc=0)
plt.xlabel('kernel size')
plt.ylabel('speed up (X)')


#%% sparse convolution with different source image size, kernel size = 6X6X1
# source image size
ssize = np.array([12,24,48,96,192])

# naive and MKL method user time
ntime = np.array([0.13/10000, 1.12/10000, 0.53/1000, 2.44/1000, 1.03/100])
dtime = np.array([0.08/10000, 0.38/10000, 0.18/1000, 0.75/1000, 0.29/100])
ftime = np.array([1.88/10000, 5.56/10000, 1.65/1000, 6.08/1000, 3.65/100])
atime = np.array([1.89/10000, 5.56/10000, 1.69/1000, 5.98/1000, 3.60/100])

# speed up
sud = ntime / dtime
suf = ntime / ftime

# plot
plt.subplot(121)
plt.plot(ssize, ntime, '-ro', label='NAIVE')
plt.plot(ssize, dtime, '-gp', label='MKL--DIRECT')
plt.plot(ssize, ftime, '-b*', label='MKL--FFT')
plt.xlabel('image size')
plt.ylabel('user time (s)')
plt.legend(loc=0)

plt.subplot(122)
plt.plot(ssize, sud, '-gp', label='DIRECT')
plt.plot(ssize, suf, '-b*', label='FFT')
plt.legend(loc=0)
plt.xlabel('image size')
plt.ylabel('speed up (X)')