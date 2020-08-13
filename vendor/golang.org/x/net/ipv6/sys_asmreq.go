// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris windows

package ipv6

import (
	"net"
	"unsafe"

	"golang.org/x/net/internal/socket"
)

func (so *sockOpt) setIPMreq(c *socket.Conn, ifi *net.Interface, grp net.IP) error {
	var mreq ipv6Mreq
	copy(mreq.Multiaddr[:], grp)
	if ifi != nil {
		mreq.setIfindex(ifi.Index)
	}
	b := (*[sizeofIPv6Mreq]byte)(unsafe.Pointer(&mreq))[:sizeofIPv6Mreq]
	return so.Set(c, b)
}
