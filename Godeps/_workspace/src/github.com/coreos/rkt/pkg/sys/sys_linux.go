package sys

import "syscall"

func Syncfs(fd int) error {
	_, _, err := syscall.RawSyscall(SYS_SYNCFS, uintptr(fd), 0, 0)
	if err != 0 {
		return syscall.Errno(err)
	}
	return nil
}
