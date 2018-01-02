// +build linux

package sys

import "golang.org/x/sys/unix"

// EpollCreate1 directly calls unix.EpollCreate1
func EpollCreate1(flag int) (int, error) {
	return unix.EpollCreate1(flag)
}

// EpollCtl directly calls unix.EpollCtl
func EpollCtl(epfd int, op int, fd int, event *unix.EpollEvent) error {
	return unix.EpollCtl(epfd, op, fd, event)
}

// EpollWait directly calls unix.EpollWait
func EpollWait(epfd int, events []unix.EpollEvent, msec int) (int, error) {
	return unix.EpollWait(epfd, events, msec)
}
