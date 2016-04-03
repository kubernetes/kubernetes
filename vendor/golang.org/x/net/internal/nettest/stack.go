// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package nettest provides utilities for IP testing.
package nettest // import "golang.org/x/net/internal/nettest"

import "net"

// SupportsIPv4 reports whether the platform supports IPv4 networking
// functionality.
func SupportsIPv4() bool {
	ln, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		return false
	}
	ln.Close()
	return true
}

// SupportsIPv6 reports whether the platform supports IPv6 networking
// functionality.
func SupportsIPv6() bool {
	ln, err := net.Listen("tcp6", "[::1]:0")
	if err != nil {
		return false
	}
	ln.Close()
	return true
}

// ProtocolNotSupported reports whether err is a protocol not
// supported error.
func ProtocolNotSupported(err error) bool {
	return protocolNotSupported(err)
}
