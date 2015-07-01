#!/bin/bash

export LD_LIBRARY_PATH=LD_LIBRARY_PATH:"/usr/local/boost/1.55.0/lib64"
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:"/opt/boost/lib"

./bin/znn --options="experiments/VeryDeep2HR_w65x9/train.config"
