#!/bin/bash

<<<<<<< HEAD
LD_LIBRARY_PATH=/usr/local/boost/1.55.0/lib64
export LD_LIBRARY_PATH
./bin/znn_amd --options="experiments/W14_C8_P3_D3/train.config"
=======
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:"/usr/local/boost/1.55.0/lib64"
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:"/opt/boost/lib"

./bin/znn --options="experiments/VeryDeep2HR_w65x9/train.config"
>>>>>>> 5b5aaec0734623575701c6a531151ee4dfe9a82c
