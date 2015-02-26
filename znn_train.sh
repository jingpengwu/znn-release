#!/bin/bash

#make

#export LD_LIBRARY_PATH=LD_LIBRARY_PATH=/usr/local/boost/1.55.0/lib64

export LD_LIBRARY_PATH="/opt/intel/composer_xe_2013_sp1.4.211/mkl/lib/intel64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/boost/1.55.0/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/fftw/gcc/openmpi-1.6.5/3.3.4/lib64/:$LD_LIBRARY_PATH"

./bin/znn_mkl --options="train.config"
