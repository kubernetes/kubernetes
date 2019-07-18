// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !darwin,!dragonfly,!freebsd,!linux,!netbsd,!openbsd,!solaris,!windows

package socket

import (
	"errors"
	"net"
	"runtime"
	"unsafe"
)

const (
	sysAF_UNSPEC = 0x0
	sysAF_INET   = 0x2
	sysAF_INET6  = 0xa

	sysSOCK_RAW = 0x3
)

func probeProtocolStack() int {
	switch runtime.GOARCH {
	case "amd64p32", "mips64p32":
		return 4
	default:
		var p uintptr
		return int(unsafe.Sizeof(p))
	}
}

func marshalInetAddr(ip net.IP, port int, zone string) []byte {
	return nil
}

func parseInetAddr(b []byte, network string) (net.Addr, error) {
	return nil, errors.New("not implemented")
}

func getsockopt(s uintptr, level, name int, b []byte) (int, error) {
	return 0, errors.New("not implemented")
}

func setsockopt(s uintptr, level, name int, b []byte) error {
	return errors.New("not implemented")
}

func recvmsg(s uintptr, h *msghdr, flags int) (int, error) {
	return 0, errors.New("not implemented")
}

func sendmsg(s uintptr, h *msghdr, flags int) (int, error) {
	return 0, errors.New("not implemented")
}

func recvmmsg(s uintptr, hs []mmsghdr, flags int) (int, error) {
	return 0, errors.New("not implemented")
}

func sendmmsg(s uintptr, hs []mmsghdr, flags int) (int, error) {
	return 0, errors.New("not implemented")
}
