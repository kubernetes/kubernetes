// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux

package ipv4

import (
	"net"
	"os"
	"unsafe"

	"golang.org/x/net/internal/iana"
)

func getsockoptIPMreqn(fd, name int) (*net.Interface, error) {
	var mreqn sysIPMreqn
	l := sysSockoptLen(sysSizeofIPMreqn)
	if err := getsockopt(fd, iana.ProtocolIP, name, unsafe.Pointer(&mreqn), &l); err != nil {
		return nil, os.NewSyscallError("getsockopt", err)
	}
	if mreqn.Ifindex == 0 {
		return nil, nil
	}
	ifi, err := net.InterfaceByIndex(int(mreqn.Ifindex))
	if err != nil {
		return nil, err
	}
	return ifi, nil
}

func setsockoptIPMreqn(fd, name int, ifi *net.Interface, grp net.IP) error {
	var mreqn sysIPMreqn
	if ifi != nil {
		mreqn.Ifindex = int32(ifi.Index)
	}
	if grp != nil {
		mreqn.Multiaddr = [4]byte{grp[0], grp[1], grp[2], grp[3]}
	}
	return os.NewSyscallError("setsockopt", setsockopt(fd, iana.ProtocolIP, name, unsafe.Pointer(&mreqn), sysSizeofIPMreqn))
}
