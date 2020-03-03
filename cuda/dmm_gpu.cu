/*
 *  dmm_gpu.cu -- Template for DMM GPU kernels
 *
 *  Copyright (C) 2019, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2019, Athena Elafrou
 */

#include "dmm.h"
#include "stdio.h"
#include <cublas_v2.h>

/*
 *  Naive kernel
 */
__global__ void dmm_gpu_naive(const value_t *A, const value_t *B, value_t *C,
                              const size_t M, const size_t N, const size_t K) {

  /* Compute the row and the column of the current thread */

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by*blockDim.y + ty;
  int col = bx*blockDim.x + tx;

  value_t sum = 0;

  /* If the thread's position is out of the array, it remains inactive */
  if (row >= M || col >= N) return;

  /* Compute the value of C */
  for (int k = 0; k < K; k++){
    sum += A[row*K+k]*B[col+k*N];
  }

  C[row*N+col] = sum;
}

/*
 *  Coalesced memory acceses of A.
 */
__global__ void dmm_gpu_coalesced_A(const value_t *A, const value_t *B,
				    value_t *C, const size_t M, const size_t N,
				    const size_t K) {

  /* Define the shared memory between the threads of the same thread block */
  __shared__ value_t A_shared[TILE_Y][TILE_X];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  /* Compute the tile of the current thread */
  int row = by * TILE_Y + ty;
  int col = bx * TILE_X + tx;

  value_t sum = 0;

  for(int m = 0; m < (K+TILE_X-1)/TILE_X; m++){
    /* Load the current tile in the shared memory and synchronize */
    A_shared[ty][tx] = A[row*K + m*TILE_X+tx];

    __syncthreads();

    for(int k = 0; k < TILE_X; k++){
      /* Compute the inner product of current tile and synchronize */
      sum += A_shared[ty][k]*B[(m*TILE_X+k)*N+col];
    }
    __syncthreads();
  }
  /* Save result */
  C[row*N+col] = sum;

}

/*
 *  Reduced memory accesses.
 */
__global__ void dmm_gpu_reduced_global(const value_t *A, const value_t *B, value_t *C,
				       const size_t M, const size_t N, const size_t K) {

  /* Define the shared memory between the threads of the same thread block */
  __shared__ value_t A_shared[TILE_Y][TILE_X];
  __shared__ value_t B_shared[TILE_Y][TILE_X];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  /* Compute the tile of the current thread */
  int row = by * TILE_Y + ty;
  int col = bx * TILE_X + tx;

  value_t sum = 0;

  for(int m = 0; m < (K+TILE_X-1)/TILE_X; m++){
    /* Load the currect tile of A and B in the shared memory and synchronize  */
    A_shared[ty][tx] = A[row*K + m*TILE_X+tx];
    B_shared[ty][tx] = B[col + (m*TILE_Y+ty)*N];

    __syncthreads();

    for(int k = 0; k < TILE_X; k++){
      /* Compute the inner product of the current tile and synchronize */
      sum += A_shared[ty][k]*B_shared[k][tx];
    }
    __syncthreads();
  }
  /* Save result */
  C[row*N+col] = sum;
}

/*
 *  Use of cuBLAS
 */
void dmm_gpu_cublas(const value_t *A, const value_t *B, value_t *C,
                    const size_t M, const size_t N, const size_t K) {
                      
  /* Define parameters for cublasSgemm */

  int lda = N;
  int ldb = K;
  int ldc = N;

  const float alf = 1;
  const float bet = 0;
  const float *alpha = &alf;
  const float *beta = &bet;

  /* Create a handle for CUBLAS */
  cublasHandle_t handle;
  cublasCreate(&handle);

  /* Compute the matrix multiplication */
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, A, lda, B, ldb, beta, C, ldc);

  /* Destroy the handle */
  cublasDestroy(handle);
}
