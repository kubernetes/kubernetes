// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.9

package ipv4

import "net"

func (c *packetHandler) readFrom(b []byte) (h *Header, p []byte, cm *ControlMessage, err error) {
	c.rawOpt.RLock()
	oob := NewControlMessage(c.rawOpt.cflags)
	c.rawOpt.RUnlock()
	n, nn, _, src, err := c.ReadMsgIP(b, oob)
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
	if nn > 0 {
		cm = new(ControlMessage)
		if err := cm.Parse(oob[:nn]); err != nil {
			return nil, nil, nil, err
		}
	}
	if src != nil && cm != nil {
		cm.Src = src.IP
	}
	return
}

func (c *packetHandler) writeTo(h *Header, p []byte, cm *ControlMessage) error {
	oob := cm.Marshal()
	wh, err := h.Marshal()
	if err != nil {
		return err
	}
	dst := new(net.IPAddr)
	if cm != nil {
		if ip := cm.Dst.To4(); ip != nil {
			dst.IP = ip
		}
	}
	if dst.IP == nil {
		dst.IP = h.Dst
	}
	wh = append(wh, p...)
	_, _, err = c.WriteMsgIP(wh, oob, dst)
	return err
}
