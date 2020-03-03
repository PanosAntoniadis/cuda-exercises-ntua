/*
 *  decls.h -- NVCC forces C++ compilation of .cu files, so we
 *             need to declare C functions using extern "C" to
 *             avoid name mangling during linkage
 *
 *  Copyright (C) 2010-2012, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2010-2012, Vasileios Karakasis
 */ 
#ifndef __DECLS_H
#define __DECLS_H

#undef  __BEGIN_C_DECLS
#undef  __END_C_DECLS

#if defined(__cplusplus) || defined(__CUDACC__)
#   define __BEGIN_C_DECLS  extern "C" {
#   define __END_C_DECLS    }
#else
#   define __BEGIN_C_DECLS
#   define __END_C_DECLS
#endif  /* __cplusplus || __CUDACC__ */

#endif  /* __DECLS_H */
