/*
 *  mat_util.c -- Dense matrix utilities.
 *
 *  Copyright (C) 2019, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2019, Athena Elafrou
 */

#include "mat_util.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void mat_init(value_t **m, const size_t M, const size_t N, const value_t val) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      m[i][j] = val;
    }
  }
}

void mat_init_rand(value_t **m, const size_t M, const size_t N,
                   const value_t max) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      m[i][j] = 2 * (((value_t)drand48()) - MAKE_VALUE_CONSTANT(.5)) * max;
    }
  }
}

bool mat_equals(const value_t *const *m1, const value_t *const *m2,
                const size_t M, const size_t N, const value_t eps) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      if (FABS(m1[i][j] - m2[i][j]) > eps){
        return false;
      }
    }
  }

  return true;
}
