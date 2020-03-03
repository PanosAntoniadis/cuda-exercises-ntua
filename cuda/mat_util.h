/*
 *  mat_util.h -- Dense matrix utilities.
 *
 *  Copyright (C) 2019, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2019, Athena Elafrou
 */

#ifndef MAT_UTIL_H
#define MAT_UTIL_H

#include "common.h"
#include <stdbool.h>

__BEGIN_C_DECLS

void mat_init(value_t **m, const size_t M, const size_t N, const value_t val);
void mat_init_rand(value_t **m, const size_t M, const size_t N,
                   const value_t max);
bool mat_equals(const value_t *const *m1, const value_t *const *m2,
                const size_t M, const size_t N, const value_t eps);

__END_C_DECLS

#endif /* MAT_UTIL_H */
