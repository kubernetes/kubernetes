// +build linux
// +build ppc64 ppc64le

package isatty

import (
	"unsafe"

	syscall "golang.org/x/sys/unix"
)

const ioctlReadTermios = syscall.TCGETS

// IsTerminal return true if the file descriptor is terminal.
func IsTerminal(fd uintptr) bool {
	var termios syscall.Termios
	_, _, err := syscall.Syscall6(syscall.SYS_IOCTL, fd, ioctlReadTermios, uintptr(unsafe.Pointer(&termios)), 0, 0, 0)
	return err == 0
}
