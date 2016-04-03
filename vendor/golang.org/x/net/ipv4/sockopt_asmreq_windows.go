// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4

import (
	"net"
	"os"
	"syscall"
	"unsafe"

	"golang.org/x/net/internal/iana"
)

func setsockoptIPMreq(fd syscall.Handle, name int, ifi *net.Interface, grp net.IP) error {
	mreq := sysIPMreq{Multiaddr: [4]byte{grp[0], grp[1], grp[2], grp[3]}}
	if err := setIPMreqInterface(&mreq, ifi); err != nil {
		return err
	}
	return os.NewSyscallError("setsockopt", syscall.Setsockopt(fd, iana.ProtocolIP, int32(name), (*byte)(unsafe.Pointer(&mreq)), int32(sysSizeofIPMreq)))
}

func getsockoptInterface(fd syscall.Handle, name int) (*net.Interface, error) {
	var b [4]byte
	l := int32(4)
	if err := syscall.Getsockopt(fd, iana.ProtocolIP, int32(name), (*byte)(unsafe.Pointer(&b[0])), &l); err != nil {
		return nil, os.NewSyscallError("getsockopt", err)
	}
	ifi, err := netIP4ToInterface(net.IPv4(b[0], b[1], b[2], b[3]))
	if err != nil {
		return nil, err
	}
	return ifi, nil
}

func setsockoptInterface(fd syscall.Handle, name int, ifi *net.Interface) error {
	ip, err := netInterfaceToIP4(ifi)
	if err != nil {
		return err
	}
	var b [4]byte
	copy(b[:], ip)
	return os.NewSyscallError("setsockopt", syscall.Setsockopt(fd, iana.ProtocolIP, int32(name), (*byte)(unsafe.Pointer(&b[0])), 4))
}
