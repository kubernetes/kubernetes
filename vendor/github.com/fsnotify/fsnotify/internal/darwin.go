//go:build darwin

package internal

import (
	"syscall"

	"golang.org/x/sys/unix"
)

var (
	ErrSyscallEACCES = syscall.EACCES
	ErrUnixEACCES    = unix.EACCES
)

var maxfiles uint64

func SetRlimit() {
	// Go 1.19 will do this automatically: https://go-review.googlesource.com/c/go/+/393354/
	var l syscall.Rlimit
	err := syscall.Getrlimit(syscall.RLIMIT_NOFILE, &l)
	if err == nil && l.Cur != l.Max {
		l.Cur = l.Max
		syscall.Setrlimit(syscall.RLIMIT_NOFILE, &l)
	}
	maxfiles = l.Cur

	if n, err := syscall.SysctlUint32("kern.maxfiles"); err == nil && uint64(n) < maxfiles {
		maxfiles = uint64(n)
	}

	if n, err := syscall.SysctlUint32("kern.maxfilesperproc"); err == nil && uint64(n) < maxfiles {
		maxfiles = uint64(n)
	}
}

func Maxfiles() uint64                              { return maxfiles }
func Mkfifo(path string, mode uint32) error         { return unix.Mkfifo(path, mode) }
func Mknod(path string, mode uint32, dev int) error { return unix.Mknod(path, mode, dev) }
