// +build !windows
// +build !linux !cgo

package term

import (
	"syscall"
	"unsafe"
)

func tcget(fd uintptr, p *Termios) syscall.Errno {
	_, _, err := syscall.Syscall(syscall.SYS_IOCTL, fd, uintptr(getTermios), uintptr(unsafe.Pointer(p)))
	return err
}

func tcset(fd uintptr, p *Termios) syscall.Errno {
	_, _, err := syscall.Syscall(syscall.SYS_IOCTL, fd, setTermios, uintptr(unsafe.Pointer(p)))
	return err
}
