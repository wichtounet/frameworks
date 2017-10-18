#!/bin/bash

#=======================================================================
# Copyright (c) 2016 Baptiste Wicht
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at
#  http://opensource.org/licenses/MIT)
#=======================================================================

# Configuration
export CAFFE_ROOT="/home/wichtounet/dev/caffe"
export TORCH_ACTIVATE="/home/wichtounet/torch/install/bin/torch-activate"
export BLAS_LIB="/opt/intel/mkl/lib/intel64/"
export PATH="$BLAS_LIB:$PATH"
export TF_ACTIVATE="/home/wichtounet/.virtualenvs/tf1.3/bin/activate"

######################
# Experiment 1 (GPU) #
######################

exp=2
mode=gpu

echo "Starting experiment 2 (GPU)"

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
export ETL_EGBLAS=true
make clean > /dev/null
make release/bin/experiment2 > /dev/null
before=`date "+%s"`
./release/bin/experiment2 | tee ../results/$exp/$mode/dll/raw_results
after=`date "+%s"`
echo "Time: $((after - before))"

# Cleanup variables
unset ETL_EGBLAS
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

source $TF_ACTIVATE

before=`date "+%s"`
CUDA_VISIBLE_DEVICES=0 python experiment2.py | tee ../results/$exp/$mode/tf/raw_results
after=`date "+%s"`
echo "Time: $((after - before))"

deactivate

cd ..

#  Keras  #
###########

echo "Starting Keras"

mkdir -p results/$exp/$mode/keras

cd keras

source ~/.virtualenvs/tf2/bin/activate

before=`date "+%s"`
CUDA_VISIBLE_DEVICES=0 python experiment2.py | tee ../results/$exp/$mode/keras/raw_results
after=`date "+%s"`
echo "Time: $((after - before))"

deactivate

cd ..

#  DeepLearning4J  #
####################

echo "Starting DeepLearning4j"

mkdir -p results/$exp/$mode/dl4j

cd dl4j

export PATH="$BLAS_LIB:$PATH"
export DL4J_MODE=cuda-8.0
mvn clean install > /dev/null

cd target/classes

before=`date "+%s"`
java -cp ../ihatejava-0.7-SNAPSHOT-bin.jar wicht.experiment2 | tee ../results/$exp/$mode/dl4j/raw_results
after=`date "+%s"`
echo "Time: $((after - before))"

cd ../..

cd ..

#  Caffe  #
###########

echo "Starting Caffe"

mkdir -p results/$exp/$mode/caffe

cd caffe

before=`date "+%s"`
$CAFFE_ROOT/build/tools/caffe train --solver=experiment2_solver_gpu.prototxt | tee ../results/$exp/$mode/caffe/raw_results
after=`date "+%s"`
echo "Time: $((after - before))"

cd ..

#  Torch  #
###########

echo "Starting Torch"

mkdir -p results/$exp/$mode/torch

cd torch

source $TORCH_ACTIVATE

before=`date "+%s"`
th experiment2_gpu.lua | tee ../results/$exp/$mode/torch/raw_results
after=`date "+%s"`
echo "Time: $((after - before))"

cd ..
