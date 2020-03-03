/*
 *  dmm.h -- Declarations and definitions related to the DMM
 *           multiplication kernels.
 *
 *  Copyright (C) 2019, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2019, Athena Elafrou
 */

#ifndef DMM_H
#define DMM_H

#include "common.h"
#include <stdbool.h>
#include <stddef.h>

// Thread block dimensions
#ifndef THREAD_BLOCK_X
#define THREAD_BLOCK_X 16
#endif

#ifndef THREAD_BLOCK_Y
#define THREAD_BLOCK_Y 16
#endif

// Tile dimensions
#ifndef TILE_X
#define TILE_X 16
#endif

#ifndef TILE_Y
#define TILE_Y 16
#endif

__BEGIN_C_DECLS

void dmm_serial(const value_t *const *A, const value_t *const *B, value_t **C,
                const size_t M, const size_t N, const size_t K);

__END_C_DECLS

#ifdef __CUDACC__
#define __MAKE_KERNEL_NAME(name) dmm_gpu##name
#define MAKE_KERNEL_NAME(name) __MAKE_KERNEL_NAME(name)

#define DECLARE_GPU_KERNEL(name)                                               \
  __global__ void MAKE_KERNEL_NAME(name)(const value_t *A, const value_t *B,   \
                                         value_t *C, const size_t M,           \
                                         const size_t N, const size_t K)

typedef void (*dmm_kernel_t)(const value_t *A, const value_t *B, value_t *C,
                             const size_t M, const size_t N, const size_t K);

typedef struct {
  const char *name;
  dmm_kernel_t fn;
} gpu_kernel_t;

enum { GPU_NAIVE = 0, GPU_COALESCED_A, GPU_REDUCED_GLOBAL, GPU_CUBLAS, GPU_KERNEL_END };

DECLARE_GPU_KERNEL(_naive);
DECLARE_GPU_KERNEL(_coalesced_A);
DECLARE_GPU_KERNEL(_reduced_global);
void dmm_gpu_cublas(const value_t *A, const value_t *B, value_t *C,
		    const size_t M, const size_t N, const size_t K);

static gpu_kernel_t gpu_kernels[] = {
    {
        .name = "naive", .fn = MAKE_KERNEL_NAME(_naive),
    },

    {
        .name = "coalesced_A", .fn = MAKE_KERNEL_NAME(_coalesced_A),
    },

    {
        .name = "reduced_global", .fn = MAKE_KERNEL_NAME(_reduced_global),
    },

    {
        .name = "cublas", .fn = dmm_gpu_cublas
    }
};

#endif /* __CUDACC__ */
#endif /* DMM_H */
