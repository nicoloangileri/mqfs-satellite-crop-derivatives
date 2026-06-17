# Build the pure-C MQFS pricing kernel.
#   make            -> libmqfs_ou_mc.so (shared, ctypes-callable) + cli demo
#   make shared     -> shared library only
#   make demo       -> standalone CLI smoke test
#   make clean
CC      ?= cc
CFLAGS  ?= -O3 -march=native -std=c99 -Wall -Wextra -fPIC
LDFLAGS ?= -lm

SO  := libmqfs_ou_mc.so
ifeq ($(shell uname),Darwin)
  SO := libmqfs_ou_mc.dylib
endif

.PHONY: all shared demo clean
all: shared demo

shared: $(SO)
$(SO): ou_mc.c ou_mc.h
	$(CC) $(CFLAGS) -shared -o $@ ou_mc.c $(LDFLAGS)

demo: cli_demo
cli_demo: cli_demo.c ou_mc.c ou_mc.h
	$(CC) $(CFLAGS) -o $@ cli_demo.c ou_mc.c $(LDFLAGS)

clean:
	rm -f $(SO) cli_demo
