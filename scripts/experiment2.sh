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

#  DLL  #
#########

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
