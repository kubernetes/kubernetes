// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd netbsd openbsd windows

package ipv4

import "net"

func setIPMreqInterface(mreq *sysIPMreq, ifi *net.Interface) error {
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
