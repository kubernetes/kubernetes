// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package ipv6

import (
	"net"

	"golang.org/x/net/internal/socket"
)

// ReadFrom reads a payload of the received IPv6 datagram, from the
// endpoint c, copying the payload into b. It returns the number of
// bytes copied into b, the control message cm and the source address
// src of the received datagram.
func (c *payloadHandler) ReadFrom(b []byte) (n int, cm *ControlMessage, src net.Addr, err error) {
	if !c.ok() {
		return 0, nil, nil, errInvalidConn
	}
	c.rawOpt.RLock()
	m := socket.Message{
		Buffers: [][]byte{b},
		OOB:     NewControlMessage(c.rawOpt.cflags),
	}
	c.rawOpt.RUnlock()
	switch c.PacketConn.(type) {
	case *net.UDPConn:
		if err := c.RecvMsg(&m, 0); err != nil {
			return 0, nil, nil, &net.OpError{Op: "read", Net: c.PacketConn.LocalAddr().Network(), Source: c.PacketConn.LocalAddr(), Err: err}
		}
	case *net.IPConn:
		if err := c.RecvMsg(&m, 0); err != nil {
			return 0, nil, nil, &net.OpError{Op: "read", Net: c.PacketConn.LocalAddr().Network(), Source: c.PacketConn.LocalAddr(), Err: err}
		}
	default:
		return 0, nil, nil, &net.OpError{Op: "read", Net: c.PacketConn.LocalAddr().Network(), Source: c.PacketConn.LocalAddr(), Err: errInvalidConnType}
	}
	if m.NN > 0 {
		cm = new(ControlMessage)
		if err := cm.Parse(m.OOB[:m.NN]); err != nil {
			return 0, nil, nil, &net.OpError{Op: "read", Net: c.PacketConn.LocalAddr().Network(), Source: c.PacketConn.LocalAddr(), Err: err}
		}
		cm.Src = netAddrToIP16(m.Addr)
	}
	return m.N, cm, m.Addr, nil
}

// WriteTo writes a payload of the IPv6 datagram, to the destination
// address dst through the endpoint c, copying the payload from b. It
// returns the number of bytes written. The control message cm allows
// the IPv6 header fields and the datagram path to be specified. The
// cm may be nil if control of the outgoing datagram is not required.
func (c *payloadHandler) WriteTo(b []byte, cm *ControlMessage, dst net.Addr) (n int, err error) {
	if !c.ok() {
		return 0, errInvalidConn
	}
	m := socket.Message{
		Buffers: [][]byte{b},
		OOB:     cm.Marshal(),
		Addr:    dst,
	}
	err = c.SendMsg(&m, 0)
	if err != nil {
		err = &net.OpError{Op: "write", Net: c.PacketConn.LocalAddr().Network(), Source: c.PacketConn.LocalAddr(), Addr: opAddr(dst), Err: err}
	}
	return m.N, err
}
