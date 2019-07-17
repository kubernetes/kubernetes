// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.9
// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package ipv6

import (
	"net"

	"golang.org/x/net/internal/socket"
)

func (c *payloadHandler) readFrom(b []byte) (int, *ControlMessage, net.Addr, error) {
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
	var cm *ControlMessage
	if m.NN > 0 {
		cm = new(ControlMessage)
		if err := cm.Parse(m.OOB[:m.NN]); err != nil {
			return 0, nil, nil, &net.OpError{Op: "read", Net: c.PacketConn.LocalAddr().Network(), Source: c.PacketConn.LocalAddr(), Err: err}
		}
		cm.Src = netAddrToIP16(m.Addr)
	}
	return m.N, cm, m.Addr, nil
}

func (c *payloadHandler) writeTo(b []byte, cm *ControlMessage, dst net.Addr) (int, error) {
	m := socket.Message{
		Buffers: [][]byte{b},
		OOB:     cm.Marshal(),
		Addr:    dst,
	}
	err := c.SendMsg(&m, 0)
	if err != nil {
		err = &net.OpError{Op: "write", Net: c.PacketConn.LocalAddr().Network(), Source: c.PacketConn.LocalAddr(), Addr: opAddr(dst), Err: err}
	}
	return m.N, err
}
