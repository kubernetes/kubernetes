// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.9

package ipv4

import (
	"net"

	"golang.org/x/net/internal/socket"
)

func (c *packetHandler) readFrom(b []byte) (h *Header, p []byte, cm *ControlMessage, err error) {
	c.rawOpt.RLock()
	m := socket.Message{
		Buffers: [][]byte{b},
		OOB:     NewControlMessage(c.rawOpt.cflags),
	}
	c.rawOpt.RUnlock()
	if err := c.RecvMsg(&m, 0); err != nil {
		return nil, nil, nil, &net.OpError{Op: "read", Net: c.IPConn.LocalAddr().Network(), Source: c.IPConn.LocalAddr(), Err: err}
	}
	var hs []byte
	if hs, p, err = slicePacket(b[:m.N]); err != nil {
		return nil, nil, nil, &net.OpError{Op: "read", Net: c.IPConn.LocalAddr().Network(), Source: c.IPConn.LocalAddr(), Err: err}
	}
	if h, err = ParseHeader(hs); err != nil {
		return nil, nil, nil, &net.OpError{Op: "read", Net: c.IPConn.LocalAddr().Network(), Source: c.IPConn.LocalAddr(), Err: err}
	}
	if m.NN > 0 {
		cm = new(ControlMessage)
		if err := cm.Parse(m.OOB[:m.NN]); err != nil {
			return nil, nil, nil, &net.OpError{Op: "read", Net: c.IPConn.LocalAddr().Network(), Source: c.IPConn.LocalAddr(), Err: err}
		}
	}
	if src, ok := m.Addr.(*net.IPAddr); ok && cm != nil {
		cm.Src = src.IP
	}
	return
}

func (c *packetHandler) writeTo(h *Header, p []byte, cm *ControlMessage) error {
	m := socket.Message{
		OOB: cm.Marshal(),
	}
	wh, err := h.Marshal()
	if err != nil {
		return err
	}
	m.Buffers = [][]byte{wh, p}
	dst := new(net.IPAddr)
	if cm != nil {
		if ip := cm.Dst.To4(); ip != nil {
			dst.IP = ip
		}
	}
	if dst.IP == nil {
		dst.IP = h.Dst
	}
	m.Addr = dst
	if err := c.SendMsg(&m, 0); err != nil {
		return &net.OpError{Op: "write", Net: c.IPConn.LocalAddr().Network(), Source: c.IPConn.LocalAddr(), Addr: opAddr(dst), Err: err}
	}
	return nil
}
