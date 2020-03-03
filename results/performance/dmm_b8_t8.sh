#!/bin/bash

#PBS -o dmm_b8_t8.out
#PBS -e dmm_b8_t8.err
#PBS -l walltime=06:00:00
#PBS -l nodes=dungani:ppn=24:cuda

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=2

gpu_kernels="0 1 2"
problem_sizes="256 512 1024 2048"
gpu_prog="./dmm_main"

cd $HOME/Lab4/cuda
echo "Benchmark started on $(date) in $(hostname)"
for i in $gpu_kernels; do
    for m in $problem_sizes; do
	for n in $problem_sizes; do
	    for k in $problem_sizes; do
                GPU_KERNEL=$i $gpu_prog $m $n $k
	    done
	done
    done
done
echo "Benchmark ended on $(date) in $(hostname)"
