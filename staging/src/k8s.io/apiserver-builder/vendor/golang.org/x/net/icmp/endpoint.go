// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package icmp

import (
	"net"
	"runtime"
	"syscall"
	"time"

	"golang.org/x/net/ipv4"
	"golang.org/x/net/ipv6"
)

var _ net.PacketConn = &PacketConn{}

// A PacketConn represents a packet network endpoint that uses either
// ICMPv4 or ICMPv6.
type PacketConn struct {
	c  net.PacketConn
	p4 *ipv4.PacketConn
	p6 *ipv6.PacketConn
}

func (c *PacketConn) ok() bool { return c != nil && c.c != nil }

// IPv4PacketConn returns the ipv4.PacketConn of c.
// It returns nil when c is not created as the endpoint for ICMPv4.
func (c *PacketConn) IPv4PacketConn() *ipv4.PacketConn {
	if !c.ok() {
		return nil
	}
	return c.p4
}

// IPv6PacketConn returns the ipv6.PacketConn of c.
// It returns nil when c is not created as the endpoint for ICMPv6.
func (c *PacketConn) IPv6PacketConn() *ipv6.PacketConn {
	if !c.ok() {
		return nil
	}
	return c.p6
}

// ReadFrom reads an ICMP message from the connection.
func (c *PacketConn) ReadFrom(b []byte) (int, net.Addr, error) {
	if !c.ok() {
		return 0, nil, syscall.EINVAL
	}
	// Please be informed that ipv4.NewPacketConn enables
	// IP_STRIPHDR option by default on Darwin.
	// See golang.org/issue/9395 for further information.
	if runtime.GOOS == "darwin" && c.p4 != nil {
		n, _, peer, err := c.p4.ReadFrom(b)
		return n, peer, err
	}
	return c.c.ReadFrom(b)
}

// WriteTo writes the ICMP message b to dst.
// Dst must be net.UDPAddr when c is a non-privileged
// datagram-oriented ICMP endpoint. Otherwise it must be net.IPAddr.
func (c *PacketConn) WriteTo(b []byte, dst net.Addr) (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	return c.c.WriteTo(b, dst)
}

// Close closes the endpoint.
func (c *PacketConn) Close() error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return c.c.Close()
}

// LocalAddr returns the local network address.
func (c *PacketConn) LocalAddr() net.Addr {
	if !c.ok() {
		return nil
	}
	return c.c.LocalAddr()
}

// SetDeadline sets the read and write deadlines associated with the
// endpoint.
func (c *PacketConn) SetDeadline(t time.Time) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return c.c.SetDeadline(t)
}

// SetReadDeadline sets the read deadline associated with the
// endpoint.
func (c *PacketConn) SetReadDeadline(t time.Time) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return c.c.SetReadDeadline(t)
}

// SetWriteDeadline sets the write deadline associated with the
// endpoint.
func (c *PacketConn) SetWriteDeadline(t time.Time) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return c.c.SetWriteDeadline(t)
}
