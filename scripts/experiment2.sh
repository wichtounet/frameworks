#!/bin/bash

#=======================================================================
# Copyright (c) 2016 Baptiste Wicht
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at
#  http://opensource.org/licenses/MIT)
#=======================================================================

################
# Experiment 2 #
################

DIR=`pwd`

echo "Starting experiment 1"
echo "In directory $DIR"

#  DLL  #
#########

echo "Starting DLL"

cd dll/

# Set variables for performance
export DLL_BLAS_PKG=mkl-threads
export ETL_MKL=true
make clean > /dev/null
make release/bin/experiment2
time ./release/bin/experiment2

# Cleanup variables
unset DLL_BLAS_PKG
unset ETL_MKL

cd ..

#  Caffe  #
###########

cd caffe

echo "Starting Caffe"

export CAFFE_ROOT="/home/wichtounet/dev/caffe-cpu"
$CAFFE_ROOT/build/tools/caffe train --solver=experiment2_solver.prototxt

cd ..
