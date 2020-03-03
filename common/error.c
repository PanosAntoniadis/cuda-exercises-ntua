/*
 *  error.c -- Error handling routines
 *
 *  Copyright (C) 2010-2012, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2010-2012, Vasileios Karakasis
 */ 
#include <errno.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "error.h"

#define ERRMSG_LEN  4096

char    *program_name = NULL;

static void
do_error(int use_errno, const char *fmt, va_list arg)
{
    char    errmsg[ERRMSG_LEN];
    
    *errmsg = '\0';   /* ensure initialization of errmsg */

    if (program_name)
        snprintf(errmsg, ERRMSG_LEN, "%s: ", program_name);

    vsnprintf(errmsg + strlen(errmsg), ERRMSG_LEN, fmt, arg);
    if (use_errno && errno)
        /* errno is set */
        snprintf(errmsg + strlen(errmsg), ERRMSG_LEN,
                 ": %s", strerror(errno));

    strncat(errmsg, "\n", ERRMSG_LEN);
    fflush(stdout);     /* in case stdout and stderr are the same */
    fputs(errmsg, stderr);
    fflush(NULL);
    return;
}

void
set_program_name(char *path)
{
    if (!program_name)
        program_name = strdup(path);
    if (!program_name)
        error(1, "strdup failed");
}

void
warning(int use_errno, const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    do_error(use_errno, fmt, ap);
    va_end(ap);
    return;
}

void
error(int use_errno, const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    do_error(use_errno, fmt, ap);
    va_end(ap);
    exit(EXIT_FAILURE);
}

void
fatal(int use_errno, const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    do_error(use_errno, fmt, ap);
    va_end(ap);
    abort();
    exit(EXIT_FAILURE);     /* should not get here */
}
