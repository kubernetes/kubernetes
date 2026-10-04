//go:build !dragonfly && !freebsd && !illumos && !linux
// +build !dragonfly,!freebsd,!illumos,!linux

package socket

import (
	"fmt"
	"runtime"

	"golang.org/x/sys/unix"
)

const sysAccept = "accept"

// accept wraps accept(2).
func accept(fd, flags int) (int, unix.Sockaddr, error) {
	if flags != 0 {
		// These operating systems have no support for flags to accept(2).
		return 0, nil, fmt.Errorf("socket: Conn.Accept flags are ineffective on %s", runtime.GOOS)
	}

	return unix.Accept(fd)
}
