#!/bin/bash

#PBS -o dmm_b32_t32.out
#PBS -e dmm_b32_t32.err
#PBS -l walltime=01:00:00
#PBS -l nodes=dungani:ppn=24:cuda

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=2

gpu_kernels="0 1 2"
gpu_prog="./dmm_main"

cd $HOME/cuda
echo "Benchmark started on $(date) in $(hostname)"
for i in $gpu_kernels; do
    GPU_KERNEL=$i $gpu_prog 2048 2048 2048
done
echo "Benchmark ended on $(date) in $(hostname)"
