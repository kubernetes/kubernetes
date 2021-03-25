// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris windows zos

package ipv4

import (
	"net"
	"unsafe"

	"golang.org/x/net/bpf"
	"golang.org/x/net/internal/socket"
)

func (so *sockOpt) getMulticastInterface(c *socket.Conn) (*net.Interface, error) {
	switch so.typ {
	case ssoTypeIPMreqn:
		return so.getIPMreqn(c)
	default:
		return so.getMulticastIf(c)
	}
}

func (so *sockOpt) setMulticastInterface(c *socket.Conn, ifi *net.Interface) error {
	switch so.typ {
	case ssoTypeIPMreqn:
		return so.setIPMreqn(c, ifi, nil)
	default:
		return so.setMulticastIf(c, ifi)
	}
}

func (so *sockOpt) getICMPFilter(c *socket.Conn) (*ICMPFilter, error) {
	b := make([]byte, so.Len)
	n, err := so.Get(c, b)
	if err != nil {
		return nil, err
	}
	if n != sizeofICMPFilter {
		return nil, errNotImplemented
	}
	return (*ICMPFilter)(unsafe.Pointer(&b[0])), nil
}

func (so *sockOpt) setICMPFilter(c *socket.Conn, f *ICMPFilter) error {
	b := (*[sizeofICMPFilter]byte)(unsafe.Pointer(f))[:sizeofICMPFilter]
	return so.Set(c, b)
}

func (so *sockOpt) setGroup(c *socket.Conn, ifi *net.Interface, grp net.IP) error {
	switch so.typ {
	case ssoTypeIPMreq:
		return so.setIPMreq(c, ifi, grp)
	case ssoTypeIPMreqn:
		return so.setIPMreqn(c, ifi, grp)
	case ssoTypeGroupReq:
		return so.setGroupReq(c, ifi, grp)
	default:
		return errNotImplemented
	}
}

func (so *sockOpt) setSourceGroup(c *socket.Conn, ifi *net.Interface, grp, src net.IP) error {
	return so.setGroupSourceReq(c, ifi, grp, src)
}

func (so *sockOpt) setBPF(c *socket.Conn, f []bpf.RawInstruction) error {
	return so.setAttachFilter(c, f)
}
