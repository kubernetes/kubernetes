// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux solaris

package ipv4

import (
	"net"
	"unsafe"

	"golang.org/x/net/internal/socket"
)

func (so *sockOpt) setGroupReq(c *socket.Conn, ifi *net.Interface, grp net.IP) error {
	var gr groupReq
	if ifi != nil {
		gr.Interface = uint32(ifi.Index)
	}
	gr.setGroup(grp)
	var b []byte
	if compatFreeBSD32 {
		var d [sizeofGroupReq + 4]byte
		s := (*[sizeofGroupReq]byte)(unsafe.Pointer(&gr))
		copy(d[:4], s[:4])
		copy(d[8:], s[4:])
		b = d[:]
	} else {
		b = (*[sizeofGroupReq]byte)(unsafe.Pointer(&gr))[:sizeofGroupReq]
	}
	return so.Set(c, b)
}

func (so *sockOpt) setGroupSourceReq(c *socket.Conn, ifi *net.Interface, grp, src net.IP) error {
	var gsr groupSourceReq
	if ifi != nil {
		gsr.Interface = uint32(ifi.Index)
	}
	gsr.setSourceGroup(grp, src)
	var b []byte
	if compatFreeBSD32 {
		var d [sizeofGroupSourceReq + 4]byte
		s := (*[sizeofGroupSourceReq]byte)(unsafe.Pointer(&gsr))
		copy(d[:4], s[:4])
		copy(d[8:], s[4:])
		b = d[:]
	} else {
		b = (*[sizeofGroupSourceReq]byte)(unsafe.Pointer(&gsr))[:sizeofGroupSourceReq]
	}
	return so.Set(c, b)
}
