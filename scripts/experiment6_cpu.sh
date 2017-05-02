#!/bin/bash

#=======================================================================
# Copyright (c) 2016 Baptiste Wicht
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at
#  http://opensource.org/licenses/MIT)
#=======================================================================

######################
# Experiment 6 (CPU) #
######################

exp=6
mode=cpu

echo "Starting experiment $exp ($mode)"

#  DLL  #
#########

echo "Starting DLL"

mkdir -p results/$exp/$mode/dll

cd dll/

# Set variables for performance
export DLL_BLAS_PKG=mkl
export ETL_MKL=true
make clean > /dev/null
make release/bin/experiment6 > /dev/null
before=`date "+%s"`
./release/bin/experiment6 | tee ../results/$exp/$mode/dll/raw_results
after=`date "+%s"`
echo "Time: $((after - before))"

# Cleanup variables
unset DLL_BLAS_PKG
unset ETL_MKL

cd ..

#  Caffe  #
###########

echo "Starting Caffe"

mkdir -p results/$exp/$mode/caffe

cd caffe

export CAFFE_ROOT="/home/wichtounet/dev/caffe-cpu"

before=`date "+%s"`
$CAFFE_ROOT/build/tools/caffe train --solver=experiment6_solver.prototxt | tee ../results/$exp/$mode/caffe/raw_results
after=`date "+%s"`
echo "Time: $((after - before))"

cd ..
