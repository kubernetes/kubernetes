//go:build !plan9 && !windows
// +build !plan9,!windows

package nltest

import "golang.org/x/sys/unix"

func isSyscallError(err error) bool {
	_, ok := err.(unix.Errno)
	return ok
}
