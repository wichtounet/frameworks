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

echo "Starting experiment 1 (CPU)"

#  DLL  #
#########

echo "Starting DLL"

cd dll/

# Set variables for performance
export DLL_BLAS_PKG=mkl-threads
export ETL_MKL=true
make clean > /dev/null
make release/bin/experiment1
time ./release/bin/experiment1

# Cleanup variables
unset DLL_BLAS_PKG
unset ETL_MKL

#  TF  #
########

cd tf

echo "Starting TensorFlow"

workon tf

CUDA_VISIBLE_DEVICES=-1 python experiment1.py

deactivate

cd ..

#  Keras  #
###########

cd keras

echo "Starting Keras"

workon tf

CUDA_VISIBLE_DEVICES=-1 python experiment1.py

deactivate

cd ..

#  Torch  #
###########

cd torch

echo "Starting Torch"

source ~/torch/install/bin/torch-activate

th experiment1.lua

cd ..

#  DeepLearning4J  #
####################

echo "Starting DeepLearning4j"

cd dl4j

export DL4J_MODE=native
mvn clean install > /dev/null

cd target/classes

java -cp ../ihatejava-0.7-SNAPSHOT-bin.jar wicht.experiment1

cd ../..

cd ..

#  Caffe  #
###########

cd caffe

echo "Starting Caffe"

export CAFFE_ROOT="/home/wichtounet/dev/caffe-cpu"
$CAFFE_ROOT/build/tools/caffe train --solver=experiment1_solver.prototxt

cd ..
