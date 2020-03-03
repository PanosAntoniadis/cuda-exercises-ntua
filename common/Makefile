.PHONY: all clean

DEBUG ?= 1
CC ?= gcc
RM = /bin/rm

MAKE_CPPFLAGS = 
MAKE_CFLAGS = -c -Wall -std=gnu99

ifeq ($(DEBUG), 1)
	MAKE_CPPFLAGS += -D_DEBUG_
	MAKE_CFLAGS += -g -O0
	MAKE_GPU_CFLAGS += -g -O0
else
	MAKE_CPPFLAGS += -DNDEBUG
	MAKE_CFLAGS += -O3
	MAKE_GPU_CFLAGS += -O3
endif

CPPFLAGS ?= $(MAKE_CPPFLAGS)
CFLAGS ?= $(MAKE_CFLAGS)
COMPILE = $(CC) $(CPPFLAGS) $(CFLAGS)

common_SOURCES = \
	alloc.c \
	error.c

common_OBJECTS = $(common_SOURCES:%.c=%.o)

%.o: %.c
	$(COMPILE) -c -o $@ $<

all: $(common_OBJECTS)

clean:	
	$(RM) -f  *.o *.s *~