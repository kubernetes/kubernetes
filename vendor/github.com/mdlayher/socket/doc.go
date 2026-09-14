// Package socket provides a low-level network connection type which integrates
// with Go's runtime network poller to provide asynchronous I/O and deadline
// support.
//
// This package focuses on UNIX-like operating systems which make use of BSD
// sockets system call APIs. It is meant to be used as a foundation for the
// creation of operating system-specific socket packages, for socket families
// such as Linux's AF_NETLINK, AF_PACKET, or AF_VSOCK. This package should not
// be used directly in end user applications.
//
// Any use of package socket should be guarded by build tags, as one would also
// use when importing the syscall or golang.org/x/sys packages.
package socket
