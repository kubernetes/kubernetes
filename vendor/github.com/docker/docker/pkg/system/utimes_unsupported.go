// +build !linux,!freebsd,!darwin

package system

import "syscall"

func LUtimesNano(path string, ts []syscall.Timespec) error {
	return ErrNotSupportedPlatform
}

func UtimesNano(path string, ts []syscall.Timespec) error {
	return ErrNotSupportedPlatform
}
