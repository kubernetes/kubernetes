// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6

import (
	"net"
	"time"

	"golang.org/x/net/internal/socket"
)

// BUG(mikio): On Windows, the JoinSourceSpecificGroup,
// LeaveSourceSpecificGroup, ExcludeSourceSpecificGroup and
// IncludeSourceSpecificGroup methods of PacketConn are not
// implemented.

// A Conn represents a network endpoint that uses IPv6 transport.
// It allows to set basic IP-level socket options such as traffic
// class and hop limit.
type Conn struct {
	genericOpt
}

type genericOpt struct {
	*socket.Conn
}

func (c *genericOpt) ok() bool { return c != nil && c.Conn != nil }

// PathMTU returns a path MTU value for the destination associated
// with the endpoint.
func (c *Conn) PathMTU() (int, error) {
	if !c.ok() {
		return 0, errInvalidConn
	}
	so, ok := sockOpts[ssoPathMTU]
	if !ok {
		return 0, errOpNoSupport
	}
	_, mtu, err := so.getMTUInfo(c.Conn)
	if err != nil {
		return 0, err
	}
	return mtu, nil
}

// NewConn returns a new Conn.
func NewConn(c net.Conn) *Conn {
	cc, _ := socket.NewConn(c)
	return &Conn{
		genericOpt: genericOpt{Conn: cc},
	}
}

// A PacketConn represents a packet network endpoint that uses IPv6
// transport. It is used to control several IP-level socket options
// including IPv6 header manipulation. It also provides datagram
// based network I/O methods specific to the IPv6 and higher layer
// protocols such as OSPF, GRE, and UDP.
type PacketConn struct {
	genericOpt
	dgramOpt
	payloadHandler
}

type dgramOpt struct {
	*socket.Conn
}

func (c *dgramOpt) ok() bool { return c != nil && c.Conn != nil }

// SetControlMessage allows to receive the per packet basis IP-level
// socket options.
func (c *PacketConn) SetControlMessage(cf ControlFlags, on bool) error {
	if !c.payloadHandler.ok() {
		return errInvalidConn
	}
	return setControlMessage(c.dgramOpt.Conn, &c.payloadHandler.rawOpt, cf, on)
}

// SetDeadline sets the read and write deadlines associated with the
// endpoint.
func (c *PacketConn) SetDeadline(t time.Time) error {
	if !c.payloadHandler.ok() {
		return errInvalidConn
	}
	return c.payloadHandler.SetDeadline(t)
}

// SetReadDeadline sets the read deadline associated with the
// endpoint.
func (c *PacketConn) SetReadDeadline(t time.Time) error {
	if !c.payloadHandler.ok() {
		return errInvalidConn
	}
	return c.payloadHandler.SetReadDeadline(t)
}

// SetWriteDeadline sets the write deadline associated with the
// endpoint.
func (c *PacketConn) SetWriteDeadline(t time.Time) error {
	if !c.payloadHandler.ok() {
		return errInvalidConn
	}
	return c.payloadHandler.SetWriteDeadline(t)
}

// Close closes the endpoint.
func (c *PacketConn) Close() error {
	if !c.payloadHandler.ok() {
		return errInvalidConn
	}
	return c.payloadHandler.Close()
}

// NewPacketConn returns a new PacketConn using c as its underlying
// transport.
func NewPacketConn(c net.PacketConn) *PacketConn {
	cc, _ := socket.NewConn(c.(net.Conn))
	return &PacketConn{
		genericOpt:     genericOpt{Conn: cc},
		dgramOpt:       dgramOpt{Conn: cc},
		payloadHandler: payloadHandler{PacketConn: c, Conn: cc},
	}
}
