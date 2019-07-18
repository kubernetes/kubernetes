// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4

import (
	"net"
	"time"

	"golang.org/x/net/internal/socket"
)

// BUG(mikio): On Windows, the JoinSourceSpecificGroup,
// LeaveSourceSpecificGroup, ExcludeSourceSpecificGroup and
// IncludeSourceSpecificGroup methods of PacketConn and RawConn are
// not implemented.

// A Conn represents a network endpoint that uses the IPv4 transport.
// It is used to control basic IP-level socket options such as TOS and
// TTL.
type Conn struct {
	genericOpt
}

type genericOpt struct {
	*socket.Conn
}

func (c *genericOpt) ok() bool { return c != nil && c.Conn != nil }

// NewConn returns a new Conn.
func NewConn(c net.Conn) *Conn {
	cc, _ := socket.NewConn(c)
	return &Conn{
		genericOpt: genericOpt{Conn: cc},
	}
}

// A PacketConn represents a packet network endpoint that uses the
// IPv4 transport. It is used to control several IP-level socket
// options including multicasting. It also provides datagram based
// network I/O methods specific to the IPv4 and higher layer protocols
// such as UDP.
type PacketConn struct {
	genericOpt
	dgramOpt
	payloadHandler
}

type dgramOpt struct {
	*socket.Conn
}

func (c *dgramOpt) ok() bool { return c != nil && c.Conn != nil }

// SetControlMessage sets the per packet IP-level socket options.
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
	return c.payloadHandler.PacketConn.SetDeadline(t)
}

// SetReadDeadline sets the read deadline associated with the
// endpoint.
func (c *PacketConn) SetReadDeadline(t time.Time) error {
	if !c.payloadHandler.ok() {
		return errInvalidConn
	}
	return c.payloadHandler.PacketConn.SetReadDeadline(t)
}

// SetWriteDeadline sets the write deadline associated with the
// endpoint.
func (c *PacketConn) SetWriteDeadline(t time.Time) error {
	if !c.payloadHandler.ok() {
		return errInvalidConn
	}
	return c.payloadHandler.PacketConn.SetWriteDeadline(t)
}

// Close closes the endpoint.
func (c *PacketConn) Close() error {
	if !c.payloadHandler.ok() {
		return errInvalidConn
	}
	return c.payloadHandler.PacketConn.Close()
}

// NewPacketConn returns a new PacketConn using c as its underlying
// transport.
func NewPacketConn(c net.PacketConn) *PacketConn {
	cc, _ := socket.NewConn(c.(net.Conn))
	p := &PacketConn{
		genericOpt:     genericOpt{Conn: cc},
		dgramOpt:       dgramOpt{Conn: cc},
		payloadHandler: payloadHandler{PacketConn: c, Conn: cc},
	}
	return p
}

// A RawConn represents a packet network endpoint that uses the IPv4
// transport. It is used to control several IP-level socket options
// including IPv4 header manipulation. It also provides datagram
// based network I/O methods specific to the IPv4 and higher layer
// protocols that handle IPv4 datagram directly such as OSPF, GRE.
type RawConn struct {
	genericOpt
	dgramOpt
	packetHandler
}

// SetControlMessage sets the per packet IP-level socket options.
func (c *RawConn) SetControlMessage(cf ControlFlags, on bool) error {
	if !c.packetHandler.ok() {
		return errInvalidConn
	}
	return setControlMessage(c.dgramOpt.Conn, &c.packetHandler.rawOpt, cf, on)
}

// SetDeadline sets the read and write deadlines associated with the
// endpoint.
func (c *RawConn) SetDeadline(t time.Time) error {
	if !c.packetHandler.ok() {
		return errInvalidConn
	}
	return c.packetHandler.IPConn.SetDeadline(t)
}

// SetReadDeadline sets the read deadline associated with the
// endpoint.
func (c *RawConn) SetReadDeadline(t time.Time) error {
	if !c.packetHandler.ok() {
		return errInvalidConn
	}
	return c.packetHandler.IPConn.SetReadDeadline(t)
}

// SetWriteDeadline sets the write deadline associated with the
// endpoint.
func (c *RawConn) SetWriteDeadline(t time.Time) error {
	if !c.packetHandler.ok() {
		return errInvalidConn
	}
	return c.packetHandler.IPConn.SetWriteDeadline(t)
}

// Close closes the endpoint.
func (c *RawConn) Close() error {
	if !c.packetHandler.ok() {
		return errInvalidConn
	}
	return c.packetHandler.IPConn.Close()
}

// NewRawConn returns a new RawConn using c as its underlying
// transport.
func NewRawConn(c net.PacketConn) (*RawConn, error) {
	cc, err := socket.NewConn(c.(net.Conn))
	if err != nil {
		return nil, err
	}
	r := &RawConn{
		genericOpt:    genericOpt{Conn: cc},
		dgramOpt:      dgramOpt{Conn: cc},
		packetHandler: packetHandler{IPConn: c.(*net.IPConn), Conn: cc},
	}
	so, ok := sockOpts[ssoHeaderPrepend]
	if !ok {
		return nil, errOpNoSupport
	}
	if err := so.SetInt(r.dgramOpt.Conn, boolint(true)); err != nil {
		return nil, err
	}
	return r, nil
}
