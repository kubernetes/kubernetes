// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !nacl,!plan9,!windows

package ipv6

import (
	"net"
	"syscall"
)

// ReadFrom reads a payload of the received IPv6 datagram, from the
// endpoint c, copying the payload into b.  It returns the number of
// bytes copied into b, the control message cm and the source address
// src of the received datagram.
func (c *payloadHandler) ReadFrom(b []byte) (n int, cm *ControlMessage, src net.Addr, err error) {
	if !c.ok() {
		return 0, nil, nil, syscall.EINVAL
	}
	oob := newControlMessage(&c.rawOpt)
	var oobn int
	switch c := c.PacketConn.(type) {
	case *net.UDPConn:
		if n, oobn, _, src, err = c.ReadMsgUDP(b, oob); err != nil {
			return 0, nil, nil, err
		}
	case *net.IPConn:
		if n, oobn, _, src, err = c.ReadMsgIP(b, oob); err != nil {
			return 0, nil, nil, err
		}
	default:
		return 0, nil, nil, errInvalidConnType
	}
	if cm, err = parseControlMessage(oob[:oobn]); err != nil {
		return 0, nil, nil, err
	}
	if cm != nil {
		cm.Src = netAddrToIP16(src)
	}
	return
}

// WriteTo writes a payload of the IPv6 datagram, to the destination
// address dst through the endpoint c, copying the payload from b.  It
// returns the number of bytes written.  The control message cm allows
// the IPv6 header fields and the datagram path to be specified.  The
// cm may be nil if control of the outgoing datagram is not required.
func (c *payloadHandler) WriteTo(b []byte, cm *ControlMessage, dst net.Addr) (n int, err error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	oob := marshalControlMessage(cm)
	if dst == nil {
		return 0, errMissingAddress
	}
	switch c := c.PacketConn.(type) {
	case *net.UDPConn:
		n, _, err = c.WriteMsgUDP(b, oob, dst.(*net.UDPAddr))
	case *net.IPConn:
		n, _, err = c.WriteMsgIP(b, oob, dst.(*net.IPAddr))
	default:
		return 0, errInvalidConnType
	}
	if err != nil {
		return 0, err
	}
	return
}
