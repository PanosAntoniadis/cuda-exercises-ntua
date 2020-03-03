SUBDIRS = common cuda

.PHONY: all clean query $(SUBDIRS)
.NOTPARALLEL: $(SUBDIRS)
.DEFAULT: all

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

clean:	$(SUBDIRS)

query:
	@echo '\n' | /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

help:
	@echo 'Mini help for using this Makefile:'; \
	echo '\tBuild release version: make DEBUG=0 [default: 1]'; \
	echo '\tDisable verification of results: make CHECK=0 [default: 1]'; \
	echo '\tDisable printing of GPU register info: make REGINFO=0 [default: 1]'; \
	echo 'You can also customly set CC, CPPFLAGS, CFLAGS and LDFLAGS as usual. Use the GPU_ prefix for gpu-specific flags.'
