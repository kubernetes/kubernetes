// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4

import (
	"errors"
	"net"
)

var (
	errOpNoSupport              = errors.New("operation not supported")
	errNoSuchInterface          = errors.New("no such interface")
	errNoSuchMulticastInterface = errors.New("no such multicast interface")
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
