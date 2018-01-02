# Dependencies

For the most part the codebase is self-contained (e.g. all dependencies are vendored), but assembly of the stage1 requires some other tools to be installed on the system.

## Build-time dependencies

### Basic

* GNU Make
* Go 1.5.3 or later
* autoconf
* aclocal (usually a part of automake)
* bash
* git
* glibc
  * development headers
  * the rkt binary links against the library
* gofmt (usually distributed with Go)
* govet (usually distributed with Go)
* TrouSerS (only when TPM is enabled)
  * development headers
  * the rkt binary links against the library
* libsystemd-journal
  * development headers
* gpg (when running functional tests)

### Additional dependencies when building any stage1 image

* glibc
  * development headers
  * the stage1 binaries link against the static library
* libdl
  * development headers
  * the stage1 binaries link against the library
* libacl
  * development headers
* C compiler

### Specific dependencies for the coreos/kvm flavor

* cat
* comm
* cpio
* gzip
* md5sum
* mktemp
* sort
* unsquashfs
* wget
* gpg (optional, required when downloading the CoreOS PXE image during the build)

### Specific dependencies for the kvm flavor

* patch
* tar
* xz
* [build dependencies for kernel][kernel-build-deps]
  * bc
  * binutils
  * openssl
* build dependencies for lkvm and/or qemu

### Specific dependencies for the src flavor

* build dependencies for systemd

## Run-time dependencies

* Linux 3.18+ (ideally 4.3+ to have overlay-on-overlay working), with the following options configured:
  * CONFIG_CGROUPS
  * CONFIG_NAMESPACES
  * CONFIG_UTS_NS
  * CONFIG_IPC_NS
  * CONFIG_PID_NS
  * CONFIG_NET_NS
  * CONFIG_OVERLAY_FS (nice to have)

### Additional run-time dependencies for all stage1 image flavors

* libacl
  * the library is optional (it is dlopened inside the stage1)

### Specific dependencies for the host flavor

* bash
* systemd >= v222
  * systemctl
  * systemd-shutdown
  * systemd
  * systemd-journald

[kernel-build-deps]: https://www.kernel.org/doc/Documentation/Changes
