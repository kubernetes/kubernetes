// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.12

package socket

import (
	"syscall"
	"unsafe"
)

//go:linkname syscall_getsockopt syscall.getsockopt
func syscall_getsockopt(s int, level int, name int, val unsafe.Pointer, vallen *uint32) error

func getsockopt(s uintptr, level, name int, b []byte) (int, error) {
	l := uint32(len(b))
	err := syscall_getsockopt(int(s), level, name, unsafe.Pointer(&b[0]), &l)
	return int(l), err
}

//go:linkname syscall_setsockopt syscall.setsockopt
func syscall_setsockopt(s int, level int, name int, val unsafe.Pointer, vallen uintptr) error

func setsockopt(s uintptr, level, name int, b []byte) error {
	return syscall_setsockopt(int(s), level, name, unsafe.Pointer(&b[0]), uintptr(len(b)))
}

//go:linkname syscall_recvmsg syscall.recvmsg
func syscall_recvmsg(s int, msg *syscall.Msghdr, flags int) (n int, err error)

func recvmsg(s uintptr, h *msghdr, flags int) (int, error) {
	return syscall_recvmsg(int(s), (*syscall.Msghdr)(unsafe.Pointer(h)), flags)
}

//go:linkname syscall_sendmsg syscall.sendmsg
func syscall_sendmsg(s int, msg *syscall.Msghdr, flags int) (n int, err error)

func sendmsg(s uintptr, h *msghdr, flags int) (int, error) {
	return syscall_sendmsg(int(s), (*syscall.Msghdr)(unsafe.Pointer(h)), flags)
}
