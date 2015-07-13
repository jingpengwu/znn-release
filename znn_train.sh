#!/bin/bash

LD_LIBRARY_PATH=/usr/local/boost/1.55.0/lib64
export LD_LIBRARY_PATH
./bin/znn_amd --options="experiments/W14_C8_P3_D3/train.config"
