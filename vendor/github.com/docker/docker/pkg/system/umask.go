// +build !windows

package system

import (
	"golang.org/x/sys/unix"
)

// Umask sets current process's file mode creation mask to newmask
// and returns oldmask.
func Umask(newmask int) (oldmask int, err error) {
	return unix.Umask(newmask), nil
}
