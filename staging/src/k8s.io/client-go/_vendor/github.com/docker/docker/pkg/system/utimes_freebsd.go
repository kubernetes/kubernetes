package system

import (
	"syscall"
	"unsafe"
)

// LUtimesNano is used to change access and modification time of the specified path.
// It's used for symbol link file because syscall.UtimesNano doesn't support a NOFOLLOW flag atm.
func LUtimesNano(path string, ts []syscall.Timespec) error {
	var _path *byte
	_path, err := syscall.BytePtrFromString(path)
	if err != nil {
		return err
	}

	if _, _, err := syscall.Syscall(syscall.SYS_LUTIMES, uintptr(unsafe.Pointer(_path)), uintptr(unsafe.Pointer(&ts[0])), 0); err != 0 && err != syscall.ENOSYS {
		return err
	}

	return nil
}
