/*
 *  gpu_util.cu -- GPU utility functions
 *
 *  Copyright (C) 2010-2013, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2010-2013, Vasileios Karakasis
 */

#include "gpu_util.h"
#include <cuda.h>

void *gpu_alloc(size_t count) {
  void *ret;
  if (cudaMalloc(&ret, count) != cudaSuccess) {
    ret = NULL;
  }

  return ret;
}

void gpu_free(void *gpuptr) { cudaFree(gpuptr); }

int copy_to_gpu(const void *host, void *gpu, size_t count) {
  if (cudaMemcpy(gpu, host, count, cudaMemcpyHostToDevice) != cudaSuccess)
    return -1;
  return 0;
}

int copy_from_gpu(void *host, const void *gpu, size_t count) {
  if (cudaMemcpy(host, gpu, count, cudaMemcpyDeviceToHost) != cudaSuccess)
    return -1;
  return 0;
}

const char *gpu_get_errmsg(cudaError_t err) { return cudaGetErrorString(err); }

const char *gpu_get_last_errmsg() { return gpu_get_errmsg(cudaGetLastError()); }
