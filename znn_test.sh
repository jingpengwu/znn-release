#!/bin/bash

export LD_LIBRARY_PATH="/opt/intel/composer_xe_2013_sp1.4.211/mkl/lib/intel64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/boost/1.55.0/lib64:$LD_LIBRARY_PATH"

./bin/znn --test_only=true --options="test.config"
