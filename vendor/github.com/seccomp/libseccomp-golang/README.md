![libseccomp Golang Bindings](https://github.com/seccomp/libseccomp-artwork/blob/main/logo/libseccomp-color_text.png)
===============================================================================
https://github.com/seccomp/libseccomp-golang

[![Go Reference](https://pkg.go.dev/badge/github.com/seccomp/libseccomp-golang.svg)](https://pkg.go.dev/github.com/seccomp/libseccomp-golang)
[![validate](https://github.com/seccomp/libseccomp-golang/actions/workflows/validate.yml/badge.svg)](https://github.com/seccomp/libseccomp-golang/actions/workflows/validate.yml)
[![test](https://github.com/seccomp/libseccomp-golang/actions/workflows/test.yml/badge.svg)](https://github.com/seccomp/libseccomp-golang/actions/workflows/test.yml)

The libseccomp library provides an easy to use, platform independent, interface
to the Linux Kernel's syscall filtering mechanism.  The libseccomp API is
designed to abstract away the underlying BPF based syscall filter language and
present a more conventional function-call based filtering interface that should
be familiar to, and easily adopted by, application developers.

The libseccomp-golang library provides a Go based interface to the libseccomp
library.

## Online Resources

The library source repository currently lives on GitHub at the following URLs:

* https://github.com/seccomp/libseccomp-golang
* https://github.com/seccomp/libseccomp

Documentation for this package is also available at:

* https://pkg.go.dev/github.com/seccomp/libseccomp-golang

## Verifying Releases

Starting with libseccomp-golang v0.10.0, the git tag corresponding to each
release should be signed by one of the libseccomp-golang maintainers.  It is
recommended that before use you verify the release tags using the following
command:

	% git tag -v <tag>

At present, only the following keys, specified via the fingerprints below, are
authorized to sign official libseccomp-golang release tags:

	Paul Moore <paul@paul-moore.com>
	7100 AADF AE6E 6E94 0D2E  0AD6 55E4 5A5A E8CA 7C8A

	Tom Hromatka <tom.hromatka@oracle.com>
	47A6 8FCE 37C7 D702 4FD6  5E11 356C E62C 2B52 4099

	Kir Kolyshkin <kolyshkin@gmail.com>
	C242 8CD7 5720 FACD CF76  B6EA 17DE 5ECB 75A1 100E

More information on GnuPG and git tag verification can be found at their
respective websites: https://git-scm.com/docs/git and https://gnupg.org.

## Installing the package

	% go get github.com/seccomp/libseccomp-golang

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
