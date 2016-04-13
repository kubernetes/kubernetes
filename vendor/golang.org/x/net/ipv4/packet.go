// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4

import (
	"net"
	"syscall"
)

// A packetHandler represents the IPv4 datagram handler.
type packetHandler struct {
	c *net.IPConn
	rawOpt
}

func (c *packetHandler) ok() bool { return c != nil && c.c != nil }

// ReadFrom reads an IPv4 datagram from the endpoint c, copying the
// datagram into b.  It returns the received datagram as the IPv4
// header h, the payload p and the control message cm.
func (c *packetHandler) ReadFrom(b []byte) (h *Header, p []byte, cm *ControlMessage, err error) {
	if !c.ok() {
		return nil, nil, nil, syscall.EINVAL
	}
	oob := newControlMessage(&c.rawOpt)
	n, oobn, _, src, err := c.c.ReadMsgIP(b, oob)
	if err != nil {
		return nil, nil, nil, err
	}
	var hs []byte
	if hs, p, err = slicePacket(b[:n]); err != nil {
		return nil, nil, nil, err
	}
	if h, err = ParseHeader(hs); err != nil {
		return nil, nil, nil, err
	}
	if cm, err = parseControlMessage(oob[:oobn]); err != nil {
		return nil, nil, nil, err
	}
	if src != nil && cm != nil {
		cm.Src = src.IP
	}
	return
}

func slicePacket(b []byte) (h, p []byte, err error) {
	if len(b) < HeaderLen {
		return nil, nil, errHeaderTooShort
	}
	hdrlen := int(b[0]&0x0f) << 2
	return b[:hdrlen], b[hdrlen:], nil
}

// WriteTo writes an IPv4 datagram through the endpoint c, copying the
// datagram from the IPv4 header h and the payload p.  The control
// message cm allows the datagram path and the outgoing interface to be
// specified.  Currently only Darwin and Linux support this.  The cm
// may be nil if control of the outgoing datagram is not required.
//
// The IPv4 header h must contain appropriate fields that include:
//
//	Version       = ipv4.Version
//	Len           = <must be specified>
//	TOS           = <must be specified>
//	TotalLen      = <must be specified>
//	ID            = platform sets an appropriate value if ID is zero
//	FragOff       = <must be specified>
//	TTL           = <must be specified>
//	Protocol      = <must be specified>
//	Checksum      = platform sets an appropriate value if Checksum is zero
//	Src           = platform sets an appropriate value if Src is nil
//	Dst           = <must be specified>
//	Options       = optional
func (c *packetHandler) WriteTo(h *Header, p []byte, cm *ControlMessage) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	oob := marshalControlMessage(cm)
	wh, err := h.Marshal()
	if err != nil {
		return err
	}
	dst := &net.IPAddr{}
	if cm != nil {
		if ip := cm.Dst.To4(); ip != nil {
			dst.IP = ip
		}
	}
	if dst.IP == nil {
		dst.IP = h.Dst
	}
	wh = append(wh, p...)
	_, _, err = c.c.WriteMsgIP(wh, oob, dst)
	return err
}
