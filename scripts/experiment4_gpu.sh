#!/bin/bash

#=======================================================================
# Copyright (c) 2016 Baptiste Wicht
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at
#  http://opensource.org/licenses/MIT)
#=======================================================================

######################
# Experiment 1 (GPU) #
######################

exp=4
mode=gpu

echo "Starting experiment 4 (GPU)"

#  DLL  #
#########

echo "Starting DLL"

mkdir -p results/$exp/$mode/dll

cd dll/

# Set variables for performance
export DLL_BLAS_PKG=mkl-threads
export ETL_MKL=true
export ETL_CUBLAS=true
export ETL_CUDNN=true
export ETL_CUFFT=true
make clean > /dev/null
make release/bin/experiment4 > /dev/null
before=`date "+%s"`
./release/bin/experiment4 | tee ../results/$exp/$mode/dll/raw_results
after=`date "+%s"`
echo "Time: $((after - before))"

# Cleanup variables
unset ETL_CUFFT
unset ETL_CUDNN
unset ETL_CUBLAS
unset ETL_MKL
unset DLL_BLAS_PKG

cd ..

#  TF  #
########

echo "Starting TensorFlow"

mkdir -p results/$exp/$mode/tf

cd tf

source ~/.virtualenvs/tf/bin/activate

before=`date "+%s"`
CUDA_VISIBLE_DEVICES=0 python experiment4.py | tee ../results/$exp/$mode/tf/raw_results
after=`date "+%s"`
echo "Time: $((after - before))"

deactivate

cd ..
