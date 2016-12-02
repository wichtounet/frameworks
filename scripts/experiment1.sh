#!/bin/bash

#=======================================================================
# Copyright (c) 2016 Baptiste Wicht
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at
#  http://opensource.org/licenses/MIT)
#=======================================================================

################
# Experiment 1 #
################

echo "Starting experiment 1"

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

#  Caffe  #
###########

cd caffe

echo "Starting Caffe"

export CAFFE_ROOT="/home/wichtounet/dev/caffe-cpu"
$CAFFE_ROOT/build/tools/caffe train --solver=experiment1_solver.prototxt

cd ..
