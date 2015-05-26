package system

import (
	"syscall"
	"unsafe"
)

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

func UtimesNano(path string, ts []syscall.Timespec) error {
	return syscall.UtimesNano(path, ts)
}
