// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4

import (
	"errors"
	"net"
)

var (
	errInvalidConn              = errors.New("invalid connection")
	errMissingAddress           = errors.New("missing address")
	errMissingHeader            = errors.New("missing header")
	errHeaderTooShort           = errors.New("header too short")
	errBufferTooShort           = errors.New("buffer too short")
	errInvalidConnType          = errors.New("invalid conn type")
	errOpNoSupport              = errors.New("operation not supported")
	errNoSuchInterface          = errors.New("no such interface")
	errNoSuchMulticastInterface = errors.New("no such multicast interface")

	// See http://www.freebsd.org/doc/en/books/porters-handbook/freebsd-versions.html.
	freebsdVersion uint32
)

func boolint(b bool) int {
	if b {
		return 1
	}
	return 0
}

func netAddrToIP4(a net.Addr) net.IP {
	switch v := a.(type) {
	case *net.UDPAddr:
		if ip := v.IP.To4(); ip != nil {
			return ip
		}
	case *net.IPAddr:
		if ip := v.IP.To4(); ip != nil {
			return ip
		}
	}
	return nil
}

func opAddr(a net.Addr) net.Addr {
	switch a.(type) {
	case *net.TCPAddr:
		if a == nil {
			return nil
		}
	case *net.UDPAddr:
		if a == nil {
			return nil
		}
	case *net.IPAddr:
		if a == nil {
			return nil
		}
	}
	return a
}
