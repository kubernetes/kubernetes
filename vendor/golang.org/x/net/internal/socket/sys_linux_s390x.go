// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package socket

import (
	"syscall"
	"unsafe"
)

const (
	sysRECVMMSG = 0x13
	sysSENDMMSG = 0x14
)

func socketcall(call, a0, a1, a2, a3, a4, a5 uintptr) (uintptr, syscall.Errno)
func rawsocketcall(call, a0, a1, a2, a3, a4, a5 uintptr) (uintptr, syscall.Errno)

func recvmmsg(s uintptr, hs []mmsghdr, flags int) (int, error) {
	n, errno := socketcall(sysRECVMMSG, s, uintptr(unsafe.Pointer(&hs[0])), uintptr(len(hs)), uintptr(flags), 0, 0)
	return int(n), errnoErr(errno)
}

func sendmmsg(s uintptr, hs []mmsghdr, flags int) (int, error) {
	n, errno := socketcall(sysSENDMMSG, s, uintptr(unsafe.Pointer(&hs[0])), uintptr(len(hs)), uintptr(flags), 0, 0)
	return int(n), errnoErr(errno)
}
