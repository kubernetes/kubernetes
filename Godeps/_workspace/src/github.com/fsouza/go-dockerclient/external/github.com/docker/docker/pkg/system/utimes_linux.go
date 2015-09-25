package system

import (
	"syscall"
	"unsafe"
)

func LUtimesNano(path string, ts []syscall.Timespec) error {
	// These are not currently available in syscall
	AT_FDCWD := -100
	AT_SYMLINK_NOFOLLOW := 0x100

	var _path *byte
	_path, err := syscall.BytePtrFromString(path)
	if err != nil {
		return err
	}

	if _, _, err := syscall.Syscall6(syscall.SYS_UTIMENSAT, uintptr(AT_FDCWD), uintptr(unsafe.Pointer(_path)), uintptr(unsafe.Pointer(&ts[0])), uintptr(AT_SYMLINK_NOFOLLOW), 0, 0); err != 0 && err != syscall.ENOSYS {
		return err
	}

	return nil
}

func UtimesNano(path string, ts []syscall.Timespec) error {
	return syscall.UtimesNano(path, ts)
}
