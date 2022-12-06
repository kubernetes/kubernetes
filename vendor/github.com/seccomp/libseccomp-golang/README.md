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

The project mailing list is currently hosted on Google Groups at the URL below,
please note that a Google account is not required to subscribe to the mailing
list.

* https://groups.google.com/d/forum/libseccomp

Documentation for this package is also available at:

* https://pkg.go.dev/github.com/seccomp/libseccomp-golang

## Installing the package

	# go get github.com/seccomp/libseccomp-golang

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
