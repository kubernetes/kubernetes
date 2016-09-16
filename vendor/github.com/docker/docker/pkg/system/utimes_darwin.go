package system

import "syscall"

// LUtimesNano is not supported by darwin platform.
func LUtimesNano(path string, ts []syscall.Timespec) error {
	return ErrNotSupportedPlatform
}
