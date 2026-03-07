//go:build dragonfly || freebsd || illumos || linux
// +build dragonfly freebsd illumos linux

package socket

import (
	"golang.org/x/sys/unix"
)

const sysAccept = "accept4"

// accept wraps accept4(2).
func accept(fd, flags int) (int, unix.Sockaddr, error) {
	return unix.Accept4(fd, flags)
}
