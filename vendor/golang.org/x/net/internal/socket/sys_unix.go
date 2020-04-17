// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build dragonfly freebsd linux,!s390x,!386 netbsd openbsd

package socket

import (
	"syscall"
	"unsafe"
)

func getsockopt(s uintptr, level, name int, b []byte) (int, error) {
	l := uint32(len(b))
	_, _, errno := syscall.Syscall6(syscall.SYS_GETSOCKOPT, s, uintptr(level), uintptr(name), uintptr(unsafe.Pointer(&b[0])), uintptr(unsafe.Pointer(&l)), 0)
	return int(l), errnoErr(errno)
}

func setsockopt(s uintptr, level, name int, b []byte) error {
	_, _, errno := syscall.Syscall6(syscall.SYS_SETSOCKOPT, s, uintptr(level), uintptr(name), uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), 0)
	return errnoErr(errno)
}

func recvmsg(s uintptr, h *msghdr, flags int) (int, error) {
	n, _, errno := syscall.Syscall(syscall.SYS_RECVMSG, s, uintptr(unsafe.Pointer(h)), uintptr(flags))
	return int(n), errnoErr(errno)
}

func sendmsg(s uintptr, h *msghdr, flags int) (int, error) {
	n, _, errno := syscall.Syscall(syscall.SYS_SENDMSG, s, uintptr(unsafe.Pointer(h)), uintptr(flags))
	return int(n), errnoErr(errno)
}
