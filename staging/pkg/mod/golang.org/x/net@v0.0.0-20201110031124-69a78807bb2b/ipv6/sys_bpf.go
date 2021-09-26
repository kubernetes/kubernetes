// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package ipv6

import (
	"unsafe"

	"golang.org/x/net/bpf"
	"golang.org/x/net/internal/socket"
	"golang.org/x/sys/unix"
)

func (so *sockOpt) setAttachFilter(c *socket.Conn, f []bpf.RawInstruction) error {
	prog := unix.SockFprog{
		Len:    uint16(len(f)),
		Filter: (*unix.SockFilter)(unsafe.Pointer(&f[0])),
	}
	b := (*[unix.SizeofSockFprog]byte)(unsafe.Pointer(&prog))[:unix.SizeofSockFprog]
	return so.Set(c, b)
}
