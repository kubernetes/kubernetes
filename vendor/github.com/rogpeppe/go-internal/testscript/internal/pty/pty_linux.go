package pty

import (
	"os"
	"strconv"
	"syscall"
	"unsafe"
)

func ptyName(f *os.File) (string, error) {
	var out uint32
	err := ioctl(f, "TIOCGPTN", syscall.TIOCGPTN, uintptr(unsafe.Pointer(&out)))
	if err != nil {
		return "", err
	}
	return "/dev/pts/" + strconv.Itoa(int(out)), nil
}

func ptyGrant(f *os.File) error {
	return nil
}

func ptyUnlock(f *os.File) error {
	var zero int
	return ioctl(f, "TIOCSPTLCK", syscall.TIOCSPTLCK, uintptr(unsafe.Pointer(&zero)))
}
