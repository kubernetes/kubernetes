// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6

import (
	"errors"
	"net"
	"runtime"
)

var (
	errInvalidConn     = errors.New("invalid connection")
	errMissingAddress  = errors.New("missing address")
	errHeaderTooShort  = errors.New("header too short")
	errInvalidConnType = errors.New("invalid conn type")
	errNotImplemented  = errors.New("not implemented on " + runtime.GOOS + "/" + runtime.GOARCH)
)

func boolint(b bool) int {
	if b {
		return 1
	}
	return 0
}

func netAddrToIP16(a net.Addr) net.IP {
	switch v := a.(type) {
	case *net.UDPAddr:
		if ip := v.IP.To16(); ip != nil && ip.To4() == nil {
			return ip
		}
	case *net.IPAddr:
		if ip := v.IP.To16(); ip != nil && ip.To4() == nil {
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
