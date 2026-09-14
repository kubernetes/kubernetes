# socket [![Test Status](https://github.com/mdlayher/socket/workflows/Test/badge.svg)](https://github.com/mdlayher/socket/actions) [![Go Reference](https://pkg.go.dev/badge/github.com/mdlayher/socket.svg)](https://pkg.go.dev/github.com/mdlayher/socket) [![Go Report Card](https://goreportcard.com/badge/github.com/mdlayher/socket)](https://goreportcard.com/report/github.com/mdlayher/socket)

Package `socket` provides a low-level network connection type which integrates
with Go's runtime network poller to provide asynchronous I/O and deadline
support. MIT Licensed.

This package focuses on UNIX-like operating systems which make use of BSD
sockets system call APIs. It is meant to be used as a foundation for the
creation of operating system-specific socket packages, for socket families such
as Linux's `AF_NETLINK`, `AF_PACKET`, or `AF_VSOCK`. This package should not be
used directly in end user applications.

Any use of package socket should be guarded by build tags, as one would also
use when importing the `syscall` or `golang.org/x/sys` packages.

## Stability

See the [CHANGELOG](./CHANGELOG.md) file for a description of changes between
releases.

This package only supports the two most recent major versions of Go, mirroring
Go's own release policy. Older versions of Go may lack critical features and bug
fixes which are necessary for this package to function correctly.
