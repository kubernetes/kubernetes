# The development version of clang is distributed as the 'clang' binary,
# while stable/released versions have a version number attached.
# Pin the default clang to a stable version.
CLANG ?= clang-17
STRIP ?= llvm-strip-17
OBJCOPY ?= llvm-objcopy-17
CFLAGS := -O2 -g -Wall -Werror $(CFLAGS)

CI_KERNEL_URL ?= https://github.com/cilium/ci-kernels/raw/master/

# Obtain an absolute path to the directory of the Makefile.
# Assume the Makefile is in the root of the repository.
REPODIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
UIDGID := $(shell stat -c '%u:%g' ${REPODIR})

# Prefer podman if installed, otherwise use docker.
# Note: Setting the var at runtime will always override.
CONTAINER_ENGINE ?= $(if $(shell command -v podman), podman, docker)
CONTAINER_RUN_ARGS ?= $(if $(filter ${CONTAINER_ENGINE}, podman), --log-driver=none, --user "${UIDGID}")

IMAGE := $(shell cat ${REPODIR}/testdata/docker/IMAGE)
VERSION := $(shell cat ${REPODIR}/testdata/docker/VERSION)

TARGETS := \
	testdata/loader-clang-11 \
	testdata/loader-clang-14 \
	testdata/loader-$(CLANG) \
	testdata/manyprogs \
	testdata/btf_map_init \
	testdata/invalid_map \
	testdata/raw_tracepoint \
	testdata/invalid_map_static \
	testdata/invalid_btf_map_init \
	testdata/strings \
	testdata/freplace \
	testdata/fentry_fexit \
	testdata/iproute2_map_compat \
	testdata/map_spin_lock \
	testdata/subprog_reloc \
	testdata/fwd_decl \
	testdata/kconfig \
	testdata/kconfig_config \
	testdata/kfunc \
	testdata/invalid-kfunc \
	testdata/kfunc-kmod \
	testdata/constants \
	testdata/errors \
	btf/testdata/relocs \
	btf/testdata/relocs_read \
	btf/testdata/relocs_read_tgt \
	btf/testdata/relocs_enum \
	cmd/bpf2go/testdata/minimal

.PHONY: all clean container-all container-shell generate

.DEFAULT_TARGET = container-all

# Build all ELF binaries using a containerized LLVM toolchain.
container-all:
	+${CONTAINER_ENGINE} run --rm -t ${CONTAINER_RUN_ARGS} \
		-v "${REPODIR}":/ebpf -w /ebpf --env MAKEFLAGS \
		--env HOME="/tmp" \
		--env BPF2GO_CC="$(CLANG)" \
		--env BPF2GO_FLAGS="-fdebug-prefix-map=/ebpf=. $(CFLAGS)" \
		"${IMAGE}:${VERSION}" \
		make all

# (debug) Drop the user into a shell inside the container as root.
# Set BPF2GO_ envs to make 'make generate' just work.
container-shell:
	${CONTAINER_ENGINE} run --rm -ti \
		-v "${REPODIR}":/ebpf -w /ebpf \
		--env BPF2GO_CC="$(CLANG)" \
		--env BPF2GO_FLAGS="-fdebug-prefix-map=/ebpf=. $(CFLAGS)" \
		"${IMAGE}:${VERSION}"

clean:
	find "$(CURDIR)" -name "*.elf" -delete
	find "$(CURDIR)" -name "*.o" -delete

format:
	find . -type f -name "*.c" | xargs clang-format -i

all: format $(addsuffix -el.elf,$(TARGETS)) $(addsuffix -eb.elf,$(TARGETS)) generate
	ln -srf testdata/loader-$(CLANG)-el.elf testdata/loader-el.elf
	ln -srf testdata/loader-$(CLANG)-eb.elf testdata/loader-eb.elf

generate:
	go generate -run "internal/cmd/gentypes" ./...
	go generate -skip "internal/cmd/gentypes" ./...

testdata/loader-%-el.elf: testdata/loader.c
	$* $(CFLAGS) -target bpfel -c $< -o $@
	$(STRIP) -g $@

testdata/loader-%-eb.elf: testdata/loader.c
	$* $(CFLAGS) -target bpfeb -c $< -o $@
	$(STRIP) -g $@

%-el.elf: %.c
	$(CLANG) $(CFLAGS) -target bpfel -c $< -o $@
	$(STRIP) -g $@

%-eb.elf : %.c
	$(CLANG) $(CFLAGS) -target bpfeb -c $< -o $@
	$(STRIP) -g $@

.PHONY: update-kernel-deps
update-kernel-deps: export KERNEL_VERSION?=6.7
update-kernel-deps:
	./testdata/sh/update-kernel-deps.sh
	$(MAKE) container-all
