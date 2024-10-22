# Get GO, GOARCH and CC values for cross-compilation.
include ../../cc_platform.mk

# List of GOARCH that nolibc supports, from:
# 	https://go.dev/doc/install/source#environment (with GOOS=linux)
#
# See nolibc supported arches in ./nolibc/arch-*.h
NOLIBC_GOARCHES := 386 amd64 arm arm64 loong64 ppc64le riscv64 s390x

ifneq (,$(filter $(GOARCH), $(NOLIBC_GOARCHES)))
	# We use the flags suggested in nolibc/nolibc.h, it makes the binary very small.
	CFLAGS += -fno-asynchronous-unwind-tables -fno-ident -s -Os -nostdlib -lgcc
else
	CFLAGS += -DRUNC_USE_STDLIB
endif

binary/runc-dmz: _dmz.c
	$(CC) $(CFLAGS) -static -o $@ $^
	$(STRIP) -gs $@
