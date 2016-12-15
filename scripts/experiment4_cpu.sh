#!/bin/bash

#=======================================================================
# Copyright (c) 2016 Baptiste Wicht
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at
#  http://opensource.org/licenses/MIT)
#=======================================================================

######################
# Experiment 1 (CPU) #
######################

exp=4
mode=cpu

echo "Starting experiment 4 (CPU)"

#  DLL  #
#########

echo "Starting DLL"

mkdir -p results/$exp/$mode/dll

cd dll/

# Set variables for performance
export DLL_BLAS_PKG=mkl
export ETL_MKL=true
make clean > /dev/null
make release/bin/experiment4 > /dev/null
before=`date "+%s"`
./release/bin/experiment4 | tee ../results/$exp/$mode/dll/raw_results
after=`date "+%s"`
echo "Time: $((after - before))"

# Cleanup variables
unset DLL_BLAS_PKG
unset ETL_MKL

cd ..

#  TF  #
########

echo "Starting TensorFlow"

mkdir -p results/$exp/$mode/tf

cd tf

source ~/.virtualenvs/tf/bin/activate

before=`date "+%s"`
CUDA_VISIBLE_DEVICES=-1 python experiment4.py | tee ../results/$exp/$mode/tf/raw_results
after=`date "+%s"`
echo "Time: $((after - before))"

deactivate

cd ..
