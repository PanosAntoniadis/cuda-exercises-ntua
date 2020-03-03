/*
 *  alloc.c -- 2D array allocation.
 *
 *  Copyright (C) 2010-2012, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2010-2012, Vasileios Karakasis
 */ 
#include <string.h>
#include <stdlib.h>
#include "alloc.h"

void **calloc_2d(size_t n, size_t m, size_t size)
{
    char    **ret = (char **) malloc(n*sizeof(char *));
    if (ret) {
        char    *area = (char *) calloc(n*m, size);
        if (area) {
            for (size_t i = 0; i < n; ++i)
                ret[i] = (char *) &area[i*m*size];
        } else {
            free(ret);
            ret = NULL;
        }
    }

    return (void **) ret;
}

void **copy_2d(void **dst, const void **src, size_t n, size_t m, size_t size)
{
    memcpy(dst[0], src[0], n*m*size);
    return dst;
}

void free_2d(void **array)
{
    free(array[0]);
    free(array);
}
