// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux

package ipv4

import (
	"net"
	"unsafe"

	"golang.org/x/net/internal/socket"
)

func (so *sockOpt) getIPMreqn(c *socket.Conn) (*net.Interface, error) {
	b := make([]byte, so.Len)
	if _, err := so.Get(c, b); err != nil {
		return nil, err
	}
	mreqn := (*ipMreqn)(unsafe.Pointer(&b[0]))
	if mreqn.Ifindex == 0 {
		return nil, nil
	}
	ifi, err := net.InterfaceByIndex(int(mreqn.Ifindex))
	if err != nil {
		return nil, err
	}
	return ifi, nil
}

func (so *sockOpt) setIPMreqn(c *socket.Conn, ifi *net.Interface, grp net.IP) error {
	var mreqn ipMreqn
	if ifi != nil {
		mreqn.Ifindex = int32(ifi.Index)
	}
	if grp != nil {
		mreqn.Multiaddr = [4]byte{grp[0], grp[1], grp[2], grp[3]}
	}
	b := (*[sizeofIPMreqn]byte)(unsafe.Pointer(&mreqn))[:sizeofIPMreqn]
	return so.Set(c, b)
}
