.PHONY: all clean query
.DEFAULT: all

DEBUG ?= 1
CHECK ?= 1
REGINFO ?= 1
CC ?= gcc
CUDA_PREFIX = /usr/local/cuda
GPU_CC ?= $(CUDA_PREFIX)/bin/nvcc
RM = /bin/rm

# Fermi GPUs in termis
nvcc_backend = gcc
nvcc_arch = -arch sm_35

MAKE_CPPFLAGS = -D__FLOAT_VALUES -DEPS=1e-4 -I../common
MAKE_CFLAGS = -c -Wall -std=gnu99
MAKE_GPU_CPPFLAGS = -I$(CUDA_PREFIX)/include
MAKE_GPU_CFLAGS = -c -ccbin $(nvcc_backend) $(nvcc_arch)

ifdef THREAD_BLOCK_X
	MAKE_CPPFLAGS += -DTHREAD_BLOCK_X=$(THREAD_BLOCK_X)
endif
ifdef THREAD_BLOCK_Y
	MAKE_CPPFLAGS += -DTHREAD_BLOCK_Y=$(THREAD_BLOCK_Y)
endif
ifdef TILE_X
	MAKE_CPPFLAGS += -DTILE_X=$(TILE_X)
endif
ifdef TILE_Y
	MAKE_CPPFLAGS += -DTILE_Y=$(TILE_Y)
endif

ifeq ($(DEBUG), 1)
	MAKE_CPPFLAGS += -D_DEBUG_
	MAKE_CFLAGS += -g -O0
	MAKE_GPU_CFLAGS += -g -O0
else
	MAKE_CPPFLAGS += -DNDEBUG
	MAKE_CFLAGS += -O3
	MAKE_GPU_CFLAGS += -O3
endif

ifeq ($(CHECK), 1)
	MAKE_GPU_CPPFLAGS += -D_CHECK_
endif

ifeq ($(REGINFO), 1)
	MAKE_GPU_CFLAGS += --ptxas-options=-v
endif

CPPFLAGS ?= $(MAKE_CPPFLAGS)
CFLAGS ?= $(MAKE_CFLAGS)
LDFLAGS ?= -L$(CUDA_PREFIX)/lib64 -lcudart -lcublas -lm
GPU_CPPFLAGS ?= $(MAKE_GPU_CPPFLAGS)
GPU_CFLAGS ?= $(MAKE_GPU_CFLAGS)

CPU_COMPILE = $(CC) $(CPPFLAGS) $(CFLAGS)
GPU_COMPILE = $(GPU_CC) $(CPPFLAGS) $(GPU_CPPFLAGS) $(GPU_CFLAGS)

PROGRAMS = dmm_main

dep_OBJECTS = $(patsubst %.o, ../common/%.o, alloc.o error.o)
dmm_OBJECTS = dmm.o dmm_gpu.o dmm_main.o gpu_util.o mat_util.o

all: $(PROGRAMS)

dmm_main: $(dmm_OBJECTS) $(dep_OBJECTS)
	$(GPU_CC) -o $@ -L../common $(dep_OBJECTS) $(dmm_OBJECTS) $(LDFLAGS)

gpu_util.o: gpu_util.cu
	$(GPU_COMPILE) -o $@ $<

dmm_gpu.o: dmm_gpu.cu
	$(GPU_COMPILE) -o $@ $<

dmm_main.o: dmm_main.cu
	$(GPU_COMPILE) -o $@ $<

%.o: %.c
	$(CPU_COMPILE) -o $@ $<

all: $(PROGRAMS)

%.o: %.h
dmm_gpu.o: dmm.h
dmm_main.o: ../common/alloc.h dmm.h ../common/error.h gpu_util.h mat_util.h

clean:
	$(RM) -f $(PROGRAMS) $(dmm_OBJECTS)

query:
	@echo '\n' | $(CUDA_PREFIX)/samples/C/bin/linux/release/deviceQuery

help:
	@echo 'Mini help for using this Makefile:'; \
	echo '\tBuild release version: make DEBUG=0 [default: 1]'; \
	echo '\tDisable building of the serial kernel: make SERIAL=0 [default: 1]'; \
	echo '\tDisable building of the GPU kernels: make GPU=0 [default: 1]'; \
	echo '\tDisable printing of GPU register info: make REGINFO=0 [default: 1]'; \
	echo 'You can also customly set CC, CPPFLAGS, CFLAGS and LDFLAGS as usual. Use the GPU_ prefix for gpu-specific flags.'; \
	echo 'NOTE: Disabling the serial kernel, also disables results checking. Consider avoiding this feature when debugging.'
