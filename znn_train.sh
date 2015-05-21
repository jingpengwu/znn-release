#!/bin/bash

LD_LIBRARY_PATH=/usr/local/boost/1.55.0/lib64
export LD_LIBRARY_PATH
./bin/znn --options="experiments/VGG_L6/train.config"
