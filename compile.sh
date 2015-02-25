#!/bin/bash

# set enviroment
module load intel
module list
source /opt/intel/composer_xe_2013_sp1.4.211/bin/compilervars.sh intel64

# compile
make mkl -j 32

# compile mkl_test
#make mkl_test -j 32
