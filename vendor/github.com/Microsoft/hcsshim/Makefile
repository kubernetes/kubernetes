BASE:=base.tar.gz
DEV_BUILD:=0

GO:=go
GO_FLAGS:=-ldflags "-s -w" # strip Go binaries
CGO_ENABLED:=0
GOMODVENDOR:=

CFLAGS:=-O2 -Wall
LDFLAGS:=-static -s # strip C binaries

GO_FLAGS_EXTRA:=
ifeq "$(GOMODVENDOR)" "1"
GO_FLAGS_EXTRA += -mod=vendor
endif
GO_BUILD_TAGS:=
ifneq ($(strip $(GO_BUILD_TAGS)),)
GO_FLAGS_EXTRA += -tags="$(GO_BUILD_TAGS)"
endif
GO_BUILD:=CGO_ENABLED=$(CGO_ENABLED) $(GO) build $(GO_FLAGS) $(GO_FLAGS_EXTRA)

SRCROOT=$(dir $(abspath $(firstword $(MAKEFILE_LIST))))
# additional directories to search for rule prerequisites and targets
VPATH=$(SRCROOT)

DELTA_TARGET=out/delta.tar.gz

ifeq "$(DEV_BUILD)" "1"
DELTA_TARGET=out/delta-dev.tar.gz
endif

ifeq "$(SNP_BUILD)" "1"
DELTA_TARGET=out/delta-snp.tar.gz
endif

# The link aliases for gcstools
GCS_TOOLS=\
	generichook \
	install-drivers

# Common path prefix.
PATH_PREFIX:=
# These have PATH_PREFIX prepended to obtain the full path in recipies e.g. $(PATH_PREFIX)/$(VMGS_TOOL)
VMGS_TOOL:=
IGVM_TOOL:=
KERNEL_PATH:=

.PHONY: all always rootfs test snp simple

.DEFAULT_GOAL := all

all: out/initrd.img out/rootfs.tar.gz

clean:
	find -name '*.o' -print0 | xargs -0 -r rm
	rm -rf bin deps rootfs out

test:
	cd $(SRCROOT) && $(GO) test -v ./internal/guest/...

rootfs: out/rootfs.vhd

snp: out/kernelinitrd.vmgs out/rootfs.hash.vhd out/rootfs.vhd out/v2056.vmgs

simple: out/simple.vmgs snp

%.vmgs: %.bin
	rm -f $@
	# du -BM returns the size of the bin file in M, eg 7M. The sed command replaces the M with *1024*1024 and then bc does the math to convert to bytes
	$(PATH_PREFIX)/$(VMGS_TOOL) create --filepath $@ --filesize `du -BM $< | sed  "s/M.*/*1024*1024/" | bc`
	$(PATH_PREFIX)/$(VMGS_TOOL) write --filepath $@ --datapath $< -i=8

# Simplest debug UVM used to test changes to the linux kernel. No dmverity protection. Boots an initramdisk rather than directly booting a vhd disk.
out/simple.bin: out/initrd.img $(PATH_PREFIX)/$(KERNEL_PATH) boot/startup_simple.sh
	rm -f $@
	python3 $(PATH_PREFIX)/$(IGVM_TOOL) -o $@ -kernel $(PATH_PREFIX)/$(KERNEL_PATH) -append "8250_core.nr_uarts=0 panic=-1 debug loglevel=7 rdinit=/startup_simple.sh" -rdinit out/initrd.img -vtl 0

ROOTFS_DEVICE:=/dev/sda
VERITY_DEVICE:=/dev/sdb
# Debug build for use with uvmtester. UVM with dm-verity protected vhd disk mounted directly via the kernel command line. Ignores corruption in dm-verity protected disk. (Use dmesg to see if dm-verity is ignoring data corruption.)
out/v2056.bin: out/rootfs.vhd out/rootfs.hash.vhd $(PATH_PREFIX)/$(KERNEL_PATH) out/rootfs.hash.datasectors out/rootfs.hash.datablocksize out/rootfs.hash.hashblocksize out/rootfs.hash.datablocks out/rootfs.hash.rootdigest out/rootfs.hash.salt boot/startup_v2056.sh
	rm -f $@
	python3 $(PATH_PREFIX)/$(IGVM_TOOL) -o $@ -kernel $(PATH_PREFIX)/$(KERNEL_PATH) -append "8250_core.nr_uarts=0 panic=-1 debug loglevel=7 root=/dev/dm-0 dm-mod.create=\"dmverity,,,ro,0 $(shell cat out/rootfs.hash.datasectors) verity 1 $(ROOTFS_DEVICE) $(VERITY_DEVICE) $(shell cat out/rootfs.hash.datablocksize) $(shell cat out/rootfs.hash.hashblocksize) $(shell cat out/rootfs.hash.datablocks) 0 sha256 $(shell cat out/rootfs.hash.rootdigest) $(shell cat out/rootfs.hash.salt) 1 ignore_corruption\" init=/startup_v2056.sh"  -vtl 0

# Full UVM with dm-verity protected vhd disk mounted directly via the kernel command line.
out/kernelinitrd.bin: out/rootfs.vhd out/rootfs.hash.vhd out/rootfs.hash.datasectors out/rootfs.hash.datablocksize out/rootfs.hash.hashblocksize out/rootfs.hash.datablocks out/rootfs.hash.rootdigest out/rootfs.hash.salt $(PATH_PREFIX)/$(KERNEL_PATH) boot/startup.sh
	rm -f $@
	python3 $(PATH_PREFIX)/$(IGVM_TOOL) -o $@ -kernel $(PATH_PREFIX)/$(KERNEL_PATH) -append "8250_core.nr_uarts=0 panic=-1 debug loglevel=7 root=/dev/dm-0 dm-mod.create=\"dmverity,,,ro,0 $(shell cat out/rootfs.hash.datasectors) verity 1 $(ROOTFS_DEVICE) $(VERITY_DEVICE) $(shell cat out/rootfs.hash.datablocksize) $(shell cat out/rootfs.hash.hashblocksize) $(shell cat out/rootfs.hash.datablocks) 0 sha256 $(shell cat out/rootfs.hash.rootdigest) $(shell cat out/rootfs.hash.salt)\" init=/startup.sh"  -vtl 0

# Rule to make a vhd from a file. This is used to create the rootfs.hash.vhd from rootfs.hash.
%.vhd: % bin/cmd/tar2ext4
	./bin/cmd/tar2ext4 -only-vhd -i $< -o $@

# Rule to make a vhd from an ext4 file. This is used to create the rootfs.vhd from rootfs.ext4.
%.vhd: %.ext4 bin/cmd/tar2ext4
	./bin/cmd/tar2ext4 -only-vhd -i $< -o $@

%.hash %.hash.info %.hash.datablocks %.hash.rootdigest %hash.datablocksize %.hash.datasectors %.hash.hashblocksize: %.ext4 %.hash.salt
	veritysetup format --no-superblock --salt $(shell cat out/rootfs.hash.salt) $< $*.hash > $*.hash.info
    # Retrieve info required by dm-verity at boot time
    # Get the blocksize of rootfs
	cat $*.hash.info | awk '/^Root hash:/{ print $$3 }' > $*.hash.rootdigest
	cat $*.hash.info | awk '/^Salt:/{ print $$2 }' > $*.hash.salt
	cat $*.hash.info | awk '/^Data block size:/{ print $$4 }' > $*.hash.datablocksize
	cat $*.hash.info | awk '/^Hash block size:/{ print $$4 }' > $*.hash.hashblocksize
	cat $*.hash.info | awk '/^Data blocks:/{ print $$3 }' > $*.hash.datablocks
	echo $$(( $$(cat $*.hash.datablocks) * $$(cat $*.hash.datablocksize) / 512 )) > $*.hash.datasectors

out/rootfs.hash.salt:
	hexdump -vn32 -e'8/4 "%08X" 1 "\n"' /dev/random > $@

out/rootfs.ext4: out/rootfs.tar.gz bin/cmd/tar2ext4
	gzip -f -d ./out/rootfs.tar.gz
	./bin/cmd/tar2ext4 -i ./out/rootfs.tar -o $@

out/rootfs.tar.gz: out/initrd.img
	rm -rf rootfs-conv
	mkdir rootfs-conv
	gunzip -c out/initrd.img | (cd rootfs-conv && cpio -imd)
	tar -zcf $@ -C rootfs-conv .
	rm -rf rootfs-conv

out/initrd.img: $(BASE) $(DELTA_TARGET) $(SRCROOT)/hack/catcpio.sh
	$(SRCROOT)/hack/catcpio.sh "$(BASE)" $(DELTA_TARGET) > out/initrd.img.uncompressed
	gzip -c out/initrd.img.uncompressed > $@
	rm out/initrd.img.uncompressed

# This target includes utilities which may be useful for testing purposes.
out/delta-dev.tar.gz: out/delta.tar.gz bin/internal/tools/snp-report
	rm -rf rootfs-dev
	mkdir rootfs-dev
	tar -xzf out/delta.tar.gz -C rootfs-dev
	cp bin/internal/tools/snp-report rootfs-dev/bin/
	tar -zcf $@ -C rootfs-dev .
	rm -rf rootfs-dev

out/delta-snp.tar.gz: out/delta.tar.gz bin/internal/tools/snp-report boot/startup_v2056.sh boot/startup_simple.sh boot/startup.sh
	rm -rf rootfs-snp
	mkdir rootfs-snp
	tar -xzf out/delta.tar.gz -C rootfs-snp
	cp boot/startup_v2056.sh rootfs-snp/startup_v2056.sh
	cp boot/startup_simple.sh rootfs-snp/startup_simple.sh
	cp boot/startup.sh rootfs-snp/startup.sh
	cp bin/internal/tools/snp-report rootfs-snp/bin/
	chmod a+x rootfs-snp/startup_v2056.sh
	chmod a+x rootfs-snp/startup_simple.sh
	chmod a+x rootfs-snp/startup.sh
	tar -zcf $@ -C rootfs-snp .
	rm -rf rootfs-snp

out/delta.tar.gz: bin/init bin/vsockexec bin/cmd/gcs bin/cmd/gcstools bin/cmd/hooks/wait-paths Makefile
	@mkdir -p out
	rm -rf rootfs
	mkdir -p rootfs/bin/
	mkdir -p rootfs/info/
	cp bin/init rootfs/
	cp bin/vsockexec rootfs/bin/
	cp bin/cmd/gcs rootfs/bin/
	cp bin/cmd/gcstools rootfs/bin/
	cp bin/cmd/hooks/wait-paths rootfs/bin/
	for tool in $(GCS_TOOLS); do ln -s gcstools rootfs/bin/$$tool; done
	git -C $(SRCROOT) rev-parse HEAD > rootfs/info/gcs.commit && \
	git -C $(SRCROOT) rev-parse --abbrev-ref HEAD > rootfs/info/gcs.branch && \
	date --iso-8601=minute --utc > rootfs/info/tar.date
	$(if $(and $(realpath $(subst .tar,.testdata.json,$(BASE))), $(shell which jq)), \
		jq -r '.IMAGE_NAME' $(subst .tar,.testdata.json,$(BASE)) 2>/dev/null > rootfs/info/image.name && \
		jq -r '.DATETIME' $(subst .tar,.testdata.json,$(BASE)) 2>/dev/null > rootfs/info/build.date)
	tar -zcf $@ -C rootfs .
	rm -rf rootfs

out/containerd-shim-runhcs-v1.exe:
	GOOS=windows $(GO_BUILD) -o $@ $(SRCROOT)/cmd/containerd-shim-runhcs-v1

bin/cmd/gcs bin/cmd/gcstools bin/cmd/hooks/wait-paths bin/cmd/tar2ext4 bin/internal/tools/snp-report bin/cmd/dmverity-vhd:
	@mkdir -p $(dir $@)
	GOOS=linux $(GO_BUILD) -o $@ $(SRCROOT)/$(@:bin/%=%)

bin/vsockexec: vsockexec/vsockexec.o vsockexec/vsock.o
	@mkdir -p bin
	$(CC) $(LDFLAGS) -o $@ $^

bin/init: init/init.o vsockexec/vsock.o
	@mkdir -p bin
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<