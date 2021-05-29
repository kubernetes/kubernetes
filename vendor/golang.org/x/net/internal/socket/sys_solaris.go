// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package socket

import (
	"syscall"
	"unsafe"
)

//go:cgo_import_dynamic libc___xnet_getsockopt __xnet_getsockopt "libsocket.so"
//go:cgo_import_dynamic libc_setsockopt setsockopt "libsocket.so"
//go:cgo_import_dynamic libc___xnet_recvmsg __xnet_recvmsg "libsocket.so"
//go:cgo_import_dynamic libc___xnet_sendmsg __xnet_sendmsg "libsocket.so"

//go:linkname procGetsockopt libc___xnet_getsockopt
//go:linkname procSetsockopt libc_setsockopt
//go:linkname procRecvmsg libc___xnet_recvmsg
//go:linkname procSendmsg libc___xnet_sendmsg

var (
	procGetsockopt uintptr
	procSetsockopt uintptr
	procRecvmsg    uintptr
	procSendmsg    uintptr
)

func sysvicall6(trap, nargs, a1, a2, a3, a4, a5, a6 uintptr) (uintptr, uintptr, syscall.Errno)
func rawSysvicall6(trap, nargs, a1, a2, a3, a4, a5, a6 uintptr) (uintptr, uintptr, syscall.Errno)

func getsockopt(s uintptr, level, name int, b []byte) (int, error) {
	l := uint32(len(b))
	_, _, errno := sysvicall6(uintptr(unsafe.Pointer(&procGetsockopt)), 5, s, uintptr(level), uintptr(name), uintptr(unsafe.Pointer(&b[0])), uintptr(unsafe.Pointer(&l)), 0)
	return int(l), errnoErr(errno)
}

func setsockopt(s uintptr, level, name int, b []byte) error {
	_, _, errno := sysvicall6(uintptr(unsafe.Pointer(&procSetsockopt)), 5, s, uintptr(level), uintptr(name), uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), 0)
	return errnoErr(errno)
}

func recvmsg(s uintptr, h *msghdr, flags int) (int, error) {
	n, _, errno := sysvicall6(uintptr(unsafe.Pointer(&procRecvmsg)), 3, s, uintptr(unsafe.Pointer(h)), uintptr(flags), 0, 0, 0)
	return int(n), errnoErr(errno)
}

func sendmsg(s uintptr, h *msghdr, flags int) (int, error) {
	n, _, errno := sysvicall6(uintptr(unsafe.Pointer(&procSendmsg)), 3, s, uintptr(unsafe.Pointer(h)), uintptr(flags), 0, 0, 0)
	return int(n), errnoErr(errno)
}

func recvmmsg(s uintptr, hs []mmsghdr, flags int) (int, error) {
	return 0, errNotImplemented
}

func sendmmsg(s uintptr, hs []mmsghdr, flags int) (int, error) {
	return 0, errNotImplemented
}
