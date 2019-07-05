// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package socket

import (
	"syscall"
	"unsafe"
)

func probeProtocolStack() int { return 4 }

const (
	sysSETSOCKOPT = 0xe
	sysGETSOCKOPT = 0xf
	sysSENDMSG    = 0x10
	sysRECVMSG    = 0x11
	sysRECVMMSG   = 0x13
	sysSENDMMSG   = 0x14
)

func socketcall(call, a0, a1, a2, a3, a4, a5 uintptr) (uintptr, syscall.Errno)
func rawsocketcall(call, a0, a1, a2, a3, a4, a5 uintptr) (uintptr, syscall.Errno)

func getsockopt(s uintptr, level, name int, b []byte) (int, error) {
	l := uint32(len(b))
	_, errno := socketcall(sysGETSOCKOPT, s, uintptr(level), uintptr(name), uintptr(unsafe.Pointer(&b[0])), uintptr(unsafe.Pointer(&l)), 0)
	return int(l), errnoErr(errno)
}

func setsockopt(s uintptr, level, name int, b []byte) error {
	_, errno := socketcall(sysSETSOCKOPT, s, uintptr(level), uintptr(name), uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), 0)
	return errnoErr(errno)
}

func recvmsg(s uintptr, h *msghdr, flags int) (int, error) {
	n, errno := socketcall(sysRECVMSG, s, uintptr(unsafe.Pointer(h)), uintptr(flags), 0, 0, 0)
	return int(n), errnoErr(errno)
}

func sendmsg(s uintptr, h *msghdr, flags int) (int, error) {
	n, errno := socketcall(sysSENDMSG, s, uintptr(unsafe.Pointer(h)), uintptr(flags), 0, 0, 0)
	return int(n), errnoErr(errno)
}

func recvmmsg(s uintptr, hs []mmsghdr, flags int) (int, error) {
	n, errno := socketcall(sysRECVMMSG, s, uintptr(unsafe.Pointer(&hs[0])), uintptr(len(hs)), uintptr(flags), 0, 0)
	return int(n), errnoErr(errno)
}

func sendmmsg(s uintptr, hs []mmsghdr, flags int) (int, error) {
	n, errno := socketcall(sysSENDMMSG, s, uintptr(unsafe.Pointer(&hs[0])), uintptr(len(hs)), uintptr(flags), 0, 0)
	return int(n), errnoErr(errno)
}
