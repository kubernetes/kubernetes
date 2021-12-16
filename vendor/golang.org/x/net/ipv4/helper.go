// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4

import (
	"errors"
	"net"
	"runtime"

	"golang.org/x/net/internal/socket"
)

var (
	errInvalidConn       = errors.New("invalid connection")
	errMissingAddress    = errors.New("missing address")
	errNilHeader         = errors.New("nil header")
	errHeaderTooShort    = errors.New("header too short")
	errExtHeaderTooShort = errors.New("extension header too short")
	errInvalidConnType   = errors.New("invalid conn type")
	errNotImplemented    = errors.New("not implemented on " + runtime.GOOS + "/" + runtime.GOARCH)

	// See https://www.freebsd.org/doc/en/books/porters-handbook/versions.html.
	freebsdVersion  uint32
	compatFreeBSD32 bool // 386 emulation on amd64
)

// See golang.org/issue/30899.
func adjustFreeBSD32(m *socket.Message) {
	// FreeBSD 12.0-RELEASE is affected by https://bugs.freebsd.org/bugzilla/show_bug.cgi?id=236737
	if 1200086 <= freebsdVersion && freebsdVersion < 1201000 {
		l := (m.NN + 4 - 1) &^ (4 - 1)
		if m.NN < l && l <= len(m.OOB) {
			m.NN = l
		}
	}
}

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
