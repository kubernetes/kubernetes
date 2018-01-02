// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x11driver

import (
	"fmt"
	"syscall"
	"unsafe"
)

// These constants are from /usr/include/linux/ipc.h
const (
	ipcPrivate = 0
	ipcCreat   = 0x1000
	ipcRmID    = 0
)

func shmOpen(size int) (shmid uintptr, addr unsafe.Pointer, err error) {
	shmid, _, errno0 := syscall.RawSyscall(syscall.SYS_SHMGET, ipcPrivate, uintptr(size), ipcCreat|0600)
	if errno0 != 0 {
		return 0, unsafe.Pointer(uintptr(0)), fmt.Errorf("shmget: %v", errno0)
	}
	p, _, errno1 := syscall.RawSyscall(syscall.SYS_SHMAT, shmid, 0, 0)
	_, _, errno2 := syscall.RawSyscall(syscall.SYS_SHMCTL, shmid, ipcRmID, 0)
	if errno1 != 0 {
		return 0, unsafe.Pointer(uintptr(0)), fmt.Errorf("shmat: %v", errno1)
	}
	if errno2 != 0 {
		return 0, unsafe.Pointer(uintptr(0)), fmt.Errorf("shmctl: %v", errno2)
	}
	return shmid, unsafe.Pointer(p), nil
}

func shmClose(p unsafe.Pointer) error {
	_, _, errno := syscall.RawSyscall(syscall.SYS_SHMDT, uintptr(p), 0, 0)
	if errno != 0 {
		return fmt.Errorf("shmdt: %v", errno)
	}
	return nil
}
