// +build !windows

package system

import (
	"syscall"
)

// Umask sets current process's file mode creation mask to newmask
// and return oldmask.
func Umask(newmask int) (oldmask int, err error) {
	return syscall.Umask(newmask), nil
}
