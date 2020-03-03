#ifndef __ERROR_H
#define __ERROR_H

#include "decls.h"

__BEGIN_C_DECLS

extern char *program_name;
void set_program_name(char *argv0);

void warning(int errnum, const char *fmt, ...);
void error(int errnum, const char *fmt, ...);
void fatal(int errnum, const char *fmt, ...);

__END_C_DECLS

#endif  /* __ERROR_H */
