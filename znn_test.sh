#!/bin/bash

LD_LIBRARY_PATH=/usr/local/boost/1.55.0/lib64
export LD_LIBRARY_PATH
./bin/znn --test_only=true --options="experiments/W13_C12_P4_D2/test.config"
