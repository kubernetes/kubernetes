package sys

import (
	"syscall"
)

// CloseOnExec sets or clears FD_CLOEXEC flag on a file descriptor
func CloseOnExec(fd int, set bool) error {
	flag := uintptr(0)
	if set {
		flag = syscall.FD_CLOEXEC
	}
	_, _, err := syscall.RawSyscall(syscall.SYS_FCNTL, uintptr(fd), syscall.F_SETFD, flag)
	if err != 0 {
		return syscall.Errno(err)
	}
	return nil
}
