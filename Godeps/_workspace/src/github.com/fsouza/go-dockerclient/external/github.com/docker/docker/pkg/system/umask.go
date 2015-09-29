// +build !windows

package system

import (
	"syscall"
)

func Umask(newmask int) (oldmask int, err error) {
	return syscall.Umask(newmask), nil
}
