// +build linux,cgo

package loopback

import (
	"unsafe"

	"golang.org/x/sys/unix"
)

func ioctlLoopCtlGetFree(fd uintptr) (int, error) {
	index, err := unix.IoctlGetInt(int(fd), LoopCtlGetFree)
	if err != nil {
		return 0, err
	}
	return index, nil
}

func ioctlLoopSetFd(loopFd, sparseFd uintptr) error {
	return unix.IoctlSetInt(int(loopFd), LoopSetFd, int(sparseFd))
}

func ioctlLoopSetStatus64(loopFd uintptr, loopInfo *loopInfo64) error {
	if _, _, err := unix.Syscall(unix.SYS_IOCTL, loopFd, LoopSetStatus64, uintptr(unsafe.Pointer(loopInfo))); err != 0 {
		return err
	}
	return nil
}

func ioctlLoopClrFd(loopFd uintptr) error {
	if _, _, err := unix.Syscall(unix.SYS_IOCTL, loopFd, LoopClrFd, 0); err != 0 {
		return err
	}
	return nil
}

func ioctlLoopGetStatus64(loopFd uintptr) (*loopInfo64, error) {
	loopInfo := &loopInfo64{}

	if _, _, err := unix.Syscall(unix.SYS_IOCTL, loopFd, LoopGetStatus64, uintptr(unsafe.Pointer(loopInfo))); err != 0 {
		return nil, err
	}
	return loopInfo, nil
}

func ioctlLoopSetCapacity(loopFd uintptr, value int) error {
	return unix.IoctlSetInt(int(loopFd), LoopSetCapacity, value)
}
