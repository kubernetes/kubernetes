// +build linux,!arm64

package sys

import (
	"syscall"
)

// EpollCreate1 directly calls syscall.EpollCreate1
func EpollCreate1(flag int) (int, error) {
	return syscall.EpollCreate1(flag)
}

// EpollCtl directly calls syscall.EpollCtl
func EpollCtl(epfd int, op int, fd int, event *syscall.EpollEvent) error {
	return syscall.EpollCtl(epfd, op, fd, event)
}

// EpollWait directly calls syscall.EpollWait
func EpollWait(epfd int, events []syscall.EpollEvent, msec int) (int, error) {
	return syscall.EpollWait(epfd, events, msec)
}
