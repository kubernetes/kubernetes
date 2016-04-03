// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux

package ipv6

import (
	"net"
	"os"
	"unsafe"
)

var freebsd32o64 bool

func setsockoptGroupReq(fd int, opt *sockOpt, ifi *net.Interface, grp net.IP) error {
	var gr sysGroupReq
	if ifi != nil {
		gr.Interface = uint32(ifi.Index)
	}
	gr.setGroup(grp)
	var p unsafe.Pointer
	var l sysSockoptLen
	if freebsd32o64 {
		var d [sysSizeofGroupReq + 4]byte
		s := (*[sysSizeofGroupReq]byte)(unsafe.Pointer(&gr))
		copy(d[:4], s[:4])
		copy(d[8:], s[4:])
		p = unsafe.Pointer(&d[0])
		l = sysSizeofGroupReq + 4
	} else {
		p = unsafe.Pointer(&gr)
		l = sysSizeofGroupReq
	}
	return os.NewSyscallError("setsockopt", setsockopt(fd, opt.level, opt.name, p, l))
}

func setsockoptGroupSourceReq(fd int, opt *sockOpt, ifi *net.Interface, grp, src net.IP) error {
	var gsr sysGroupSourceReq
	if ifi != nil {
		gsr.Interface = uint32(ifi.Index)
	}
	gsr.setSourceGroup(grp, src)
	var p unsafe.Pointer
	var l sysSockoptLen
	if freebsd32o64 {
		var d [sysSizeofGroupSourceReq + 4]byte
		s := (*[sysSizeofGroupSourceReq]byte)(unsafe.Pointer(&gsr))
		copy(d[:4], s[:4])
		copy(d[8:], s[4:])
		p = unsafe.Pointer(&d[0])
		l = sysSizeofGroupSourceReq + 4
	} else {
		p = unsafe.Pointer(&gsr)
		l = sysSizeofGroupSourceReq
	}
	return os.NewSyscallError("setsockopt", setsockopt(fd, opt.level, opt.name, p, l))
}
