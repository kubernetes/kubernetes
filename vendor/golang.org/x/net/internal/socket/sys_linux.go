// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux,!s390x,!386

package socket

import (
	"syscall"
	"unsafe"
)

func probeProtocolStack() int {
	var p uintptr
	return int(unsafe.Sizeof(p))
}

func recvmmsg(s uintptr, hs []mmsghdr, flags int) (int, error) {
	n, _, errno := syscall.Syscall6(sysRECVMMSG, s, uintptr(unsafe.Pointer(&hs[0])), uintptr(len(hs)), uintptr(flags), 0, 0)
	return int(n), errnoErr(errno)
}

func sendmmsg(s uintptr, hs []mmsghdr, flags int) (int, error) {
	n, _, errno := syscall.Syscall6(sysSENDMMSG, s, uintptr(unsafe.Pointer(&hs[0])), uintptr(len(hs)), uintptr(flags), 0, 0)
	return int(n), errnoErr(errno)
}
