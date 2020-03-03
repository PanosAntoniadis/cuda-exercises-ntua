/*
 *  dmm_main.cu -- DMM front-end program.
 *
 *  Copyright (C) 2019, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2019, Athena Elafrou
 */

#include "alloc.h"
#include "dmm.h"
#include "error.h"
#include "gpu_util.h"
#include "mat_util.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef VALUES_MAX
#define VALUES_MAX MAKE_VALUE_CONSTANT(1.)
#endif

#ifndef EPS
#define EPS MAKE_VALUE_CONSTANT(1.e-5)
#endif

#ifndef NR_ITER
#define NR_ITER 100
#endif

static void check_result(const value_t *const *test, const value_t *const *orig,
                         const size_t M, const size_t N) {
  printf("Checking ... ");
  bool ret = mat_equals(test, orig, M, N, EPS);
  if (ret) {
    printf("PASSED!\n");
  } else {
    printf("FAILED!\n");
  }
}

static void report_results(float time, const size_t M, const size_t N,
                           const size_t K) {
  size_t flops = 2 * M * N * K * NR_ITER;

  printf("Elapsed time: %lf ms\n", time);
  printf("Performance:  %lf Gflop/s\n", flops * 1.e-6 / time);
}

static void print_usage() {
  printf("Usage: [GPU_KERNEL=<kernel_no>] %s <M> <N> <K>\n",
         program_name);
  printf("KERNEL defaults to 0\n");
  printf("Available kernels [id:name]:\n");
  size_t i;
  for (i = 0; i < GPU_KERNEL_END; ++i) {
    printf("\t%zd:%s\n", i, gpu_kernels[i].name);
  }
}

int main(int argc, char **argv) {
  set_program_name(argv[0]);
  if (argc < 4) {
    warning(0, "too few arguments");
    print_usage();
    exit(EXIT_FAILURE);
  }

  size_t M = atoi(argv[1]);
  if (!M)
    error(0, "invalid argument: %s", argv[1]);
  size_t N = atoi(argv[2]);
  if (!N)
    error(0, "invalid argument: %s", argv[2]);
  size_t K = atoi(argv[3]);
  if (!K)
    error(0, "invalid argument: %s", argv[3]);

  /* Read block size and kernel to launch from the environment */
  const char *env_gpu_kernel = getenv("GPU_KERNEL");
  int kernel = (env_gpu_kernel) ? atoi(env_gpu_kernel) : GPU_NAIVE;
  size_t orig_M = M;
  size_t orig_N = N;
  size_t orig_K = K;

  printf("Dimension M: %zd\n", orig_M);
  printf("Adjusted dimension M: %zd\n", M);
  printf("Dimension N: %zd\n", orig_N);
  printf("Adjusted dimension N: %zd\n", N);
  printf("Dimension K: %zd\n", orig_K);
  printf("Adjusted dimension K: %zd\n", K);

  /*
   * Allocate the structures.
   *
   * Initialization to zero is crucial if you adjusted the matrix
   * size.
   */
  value_t **A = (value_t **)calloc_2d(M, K, sizeof(**A));
  if (!A)
    error(1, "alloc_2d failed");

  value_t **B = (value_t **)calloc_2d(K, N, sizeof(**B));
  if (!B)
    error(1, "alloc_2d failed");

  value_t **C = (value_t **)calloc_2d(M, N, sizeof(**C));
  if (!C)
    error(1, "alloc_2d failed");

#ifdef _CHECK_
  value_t **C_serial = (value_t **)calloc_2d(M, N, sizeof(**C_serial));
  if (!C_serial)
    error(1, "alloc_2d failed");
#endif

  /* Initialize */
  srand48(0);
  mat_init_rand(A, orig_M, orig_K, VALUES_MAX);
  mat_init_rand(B, orig_K, orig_N, VALUES_MAX);


  /* Setup timers */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  dim3 gpu_block(THREAD_BLOCK_Y, THREAD_BLOCK_X);
  dim3 gpu_grid((N+THREAD_BLOCK_Y-1)/THREAD_BLOCK_Y, (M+THREAD_BLOCK_X-1)/THREAD_BLOCK_X);

  printf(">>>> Begin of record <<<<\n");
  printf("Block dimensions: %dx%d\n", gpu_block.x, gpu_block.y);
  printf("Grid dimensions : %dx%d\n", gpu_grid.x, gpu_grid.y);

  /* GPU allocations */
  value_t *gpu_A = (value_t *)gpu_alloc(M * K * sizeof(*gpu_A));
  if (!gpu_A)
    error(0, "gpu_alloc failed: %s", gpu_get_last_errmsg());

  value_t *gpu_B = (value_t *)gpu_alloc(K * N * sizeof(*gpu_B));
  if (!gpu_B)
    error(0, "gpu_alloc failed: %s", gpu_get_last_errmsg());

  value_t *gpu_C = (value_t *)gpu_alloc(M * N * sizeof(*gpu_C));
  if (!gpu_C)
    error(0, "gpu_alloc failed: %s", gpu_get_last_errmsg());

  /* Copy data to GPU */
  if (copy_to_gpu(A[0], gpu_A, M * K * sizeof(*gpu_A)) < 0)
    error(0, "copy_to_gpu failed: %s", gpu_get_last_errmsg());

  if (copy_to_gpu(B[0], gpu_B, K * N * sizeof(*gpu_B)) < 0)
    error(0, "copy_to_gpu failed: %s", gpu_get_last_errmsg());

  /* Reset C and copy it to GPU */
  mat_init(C, M, N, MAKE_VALUE_CONSTANT(0.0));
  if (copy_to_gpu(C[0], gpu_C, M * N * sizeof(*gpu_C)) < 0)
    error(0, "copy_to_gpu failed: %s", gpu_get_last_errmsg());

  if (kernel >= GPU_KERNEL_END)
    error(0, "the requested kernel does not exist");

  printf("GPU kernel version: %s\n", gpu_kernels[kernel].name);

  /* Execute and time the kernel */
  cudaEventRecord(start);
  if (kernel == GPU_CUBLAS) {
    for (size_t i = 0; i < NR_ITER; ++i)
      gpu_kernels[kernel].fn(gpu_B,
			     gpu_A,
			     gpu_C,
			     M, N, K);
  } else {
    for (size_t i = 0; i < NR_ITER; ++i)
      gpu_kernels[kernel].fn<<<gpu_grid, gpu_block>>>(gpu_A,
						      gpu_B,
						      gpu_C,
						      M, N, K);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
#ifdef _DEBUG_
  cudaError_t err;
  if ((err = cudaGetLastError()) != cudaSuccess)
    error(0, "gpu kernel failed to launch: %s", gpu_get_errmsg(err));
#endif
  float dmm_time = 0; // time in milliseconds
  cudaEventElapsedTime(&dmm_time, start, stop);

  /* Copy result back to host and check */
  if (copy_from_gpu(C[0], gpu_C, M * N * sizeof(*gpu_C)) < 0)
    error(0, "copy_from_gpu failed: %s", gpu_get_last_errmsg());

#ifdef _CHECK_
  /* Compute serial */
  dmm_serial(A, B, C_serial, orig_M, orig_N, orig_K);
  check_result(C, C_serial, orig_M, orig_N);
#endif

  report_results(dmm_time, orig_M, orig_N, orig_K);
  printf(">>>> End of record <<<<\n");

  /* Free resources on host */
  free_2d((void **)A);
  free_2d((void **)B);
  free_2d((void **)C);
#ifdef _CHECK_
  free_2d((void **)C_serial);
#endif

  /* Free resources on GPU */
  gpu_free(gpu_A);
  gpu_free(gpu_B);
  gpu_free(gpu_C);

  return EXIT_SUCCESS;
}
