//go:build !darwin
// +build !darwin

package socket

import "golang.org/x/sys/unix"

const (
	// These operating systems support CLOEXEC and NONBLOCK socket options.
	flagCLOEXEC = true
	socketFlags = unix.SOCK_CLOEXEC | unix.SOCK_NONBLOCK
)
