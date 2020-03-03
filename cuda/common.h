/*
 *  common.h -- Basic definitions and declarations.
 *
 *  Copyright (C) 2010-2012, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2010-2012, Vasileios Karakasis
 */

#ifndef COMMON_H
#define COMMON_H

#include <stddef.h>
#define USEC_PER_SEC 1000000L

#undef __BEGIN_C_DECLS
#undef __END_C_DECLS

#if defined(__cplusplus) || defined(__CUDACC__)
#define __BEGIN_C_DECLS extern "C" {
#define __END_C_DECLS }
#else
#define __BEGIN_C_DECLS
#define __END_C_DECLS
#endif /* __cplusplus || __CUDACC__ */

#if defined(__FLOAT_VALUES) || defined(__CUDACC__)
#define MAKE_VALUE_CONSTANT(v) v##f
#define VALUE_FORMAT "f"
#define FABS fabsf
typedef float value_t;
#else
#define MAKE_VALUE_CONSTANT(v) v
#define VALUE_FORMAT "lf"
#define FABS fabs
typedef double value_t;
#endif

#endif /* COMMON_H */
