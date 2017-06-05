// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6

import (
	"os"
	"unsafe"

	"golang.org/x/net/bpf"
)

// SetBPF attaches a BPF program to the connection.
//
// Only supported on Linux.
func (c *dgramOpt) SetBPF(filter []bpf.RawInstruction) error {
	fd, err := c.sysfd()
	if err != nil {
		return err
	}
	prog := sysSockFProg{
		Len:    uint16(len(filter)),
		Filter: (*sysSockFilter)(unsafe.Pointer(&filter[0])),
	}
	return os.NewSyscallError("setsockopt", setsockopt(fd, sysSOL_SOCKET, sysSO_ATTACH_FILTER, unsafe.Pointer(&prog), uint32(unsafe.Sizeof(prog))))
}
