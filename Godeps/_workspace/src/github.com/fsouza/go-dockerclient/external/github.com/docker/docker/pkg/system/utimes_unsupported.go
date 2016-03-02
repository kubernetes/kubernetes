// +build !linux,!freebsd,!darwin

package system

import "syscall"

// LUtimesNano is not supported on platforms other than linux, freebsd and darwin.
func LUtimesNano(path string, ts []syscall.Timespec) error {
	return ErrNotSupportedPlatform
}
