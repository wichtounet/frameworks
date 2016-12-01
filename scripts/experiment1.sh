#!/bin/bash

################
# Experiment 1 #
################

#  DLL  #
#########

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
