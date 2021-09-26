// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd netbsd openbsd solaris windows

package ipv4

import (
	"errors"
	"net"
	"unsafe"

	"golang.org/x/net/internal/socket"
)

var errNoSuchInterface = errors.New("no such interface")

func (so *sockOpt) setIPMreq(c *socket.Conn, ifi *net.Interface, grp net.IP) error {
	mreq := ipMreq{Multiaddr: [4]byte{grp[0], grp[1], grp[2], grp[3]}}
	if err := setIPMreqInterface(&mreq, ifi); err != nil {
		return err
	}
	b := (*[sizeofIPMreq]byte)(unsafe.Pointer(&mreq))[:sizeofIPMreq]
	return so.Set(c, b)
}

func (so *sockOpt) getMulticastIf(c *socket.Conn) (*net.Interface, error) {
	var b [4]byte
	if _, err := so.Get(c, b[:]); err != nil {
		return nil, err
	}
	ifi, err := netIP4ToInterface(net.IPv4(b[0], b[1], b[2], b[3]))
	if err != nil {
		return nil, err
	}
	return ifi, nil
}

func (so *sockOpt) setMulticastIf(c *socket.Conn, ifi *net.Interface) error {
	ip, err := netInterfaceToIP4(ifi)
	if err != nil {
		return err
	}
	var b [4]byte
	copy(b[:], ip)
	return so.Set(c, b[:])
}

func setIPMreqInterface(mreq *ipMreq, ifi *net.Interface) error {
	if ifi == nil {
		return nil
	}
	ifat, err := ifi.Addrs()
	if err != nil {
		return err
	}
	for _, ifa := range ifat {
		switch ifa := ifa.(type) {
		case *net.IPAddr:
			if ip := ifa.IP.To4(); ip != nil {
				copy(mreq.Interface[:], ip)
				return nil
			}
		case *net.IPNet:
			if ip := ifa.IP.To4(); ip != nil {
				copy(mreq.Interface[:], ip)
				return nil
			}
		}
	}
	return errNoSuchInterface
}

func netIP4ToInterface(ip net.IP) (*net.Interface, error) {
	ift, err := net.Interfaces()
	if err != nil {
		return nil, err
	}
	for _, ifi := range ift {
		ifat, err := ifi.Addrs()
		if err != nil {
			return nil, err
		}
		for _, ifa := range ifat {
			switch ifa := ifa.(type) {
			case *net.IPAddr:
				if ip.Equal(ifa.IP) {
					return &ifi, nil
				}
			case *net.IPNet:
				if ip.Equal(ifa.IP) {
					return &ifi, nil
				}
			}
		}
	}
	return nil, errNoSuchInterface
}

func netInterfaceToIP4(ifi *net.Interface) (net.IP, error) {
	if ifi == nil {
		return net.IPv4zero.To4(), nil
	}
	ifat, err := ifi.Addrs()
	if err != nil {
		return nil, err
	}
	for _, ifa := range ifat {
		switch ifa := ifa.(type) {
		case *net.IPAddr:
			if ip := ifa.IP.To4(); ip != nil {
				return ip, nil
			}
		case *net.IPNet:
			if ip := ifa.IP.To4(); ip != nil {
				return ip, nil
			}
		}
	}
	return nil, errNoSuchInterface
}
