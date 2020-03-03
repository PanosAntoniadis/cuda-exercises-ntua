/*
 *  dmm.c -- Declarations and definitions related to the DMM
 *           multiplication kernels.
 *
 *  Copyright (C) 2019, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2019, Athena Elafrou
 */

#include "dmm.h"

void dmm_serial(const value_t *const *A, const value_t *const *B, value_t **C,
                const size_t M, const size_t N, const size_t K) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      register value_t _Cij = 0;
      for (size_t k = 0; k < K; ++k) {
        _Cij += A[i][k] * B[k][j];
      }

      C[i][j] = _Cij;
    }
  }
}
