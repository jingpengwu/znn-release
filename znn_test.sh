#!/bin/bash

<<<<<<< HEAD
LD_LIBRARY_PATH=/usr/local/boost/1.55.0/lib64
export LD_LIBRARY_PATH
./bin/znn --test_only=true --options="experiments/W13_C12_P4_D2/test.config"
=======
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:"/usr/local/boost/1.55.0/lib64"
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:"/opt/boost/lib"

./bin/znn --test_only=true --options="experiments/VeryDeep2HR_w65x9/test.config"
>>>>>>> 5b5aaec0734623575701c6a531151ee4dfe9a82c
