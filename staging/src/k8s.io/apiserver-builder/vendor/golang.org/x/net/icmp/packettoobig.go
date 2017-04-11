// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package icmp

import "encoding/binary"

// A PacketTooBig represents an ICMP packet too big message body.
type PacketTooBig struct {
	MTU  int    // maximum transmission unit of the nexthop link
	Data []byte // data, known as original datagram field
}

// Len implements the Len method of MessageBody interface.
func (p *PacketTooBig) Len(proto int) int {
	if p == nil {
		return 0
	}
	return 4 + len(p.Data)
}

// Marshal implements the Marshal method of MessageBody interface.
func (p *PacketTooBig) Marshal(proto int) ([]byte, error) {
	b := make([]byte, 4+len(p.Data))
	binary.BigEndian.PutUint32(b[:4], uint32(p.MTU))
	copy(b[4:], p.Data)
	return b, nil
}

// parsePacketTooBig parses b as an ICMP packet too big message body.
func parsePacketTooBig(proto int, b []byte) (MessageBody, error) {
	bodyLen := len(b)
	if bodyLen < 4 {
		return nil, errMessageTooShort
	}
	p := &PacketTooBig{MTU: int(binary.BigEndian.Uint32(b[:4]))}
	if bodyLen > 4 {
		p.Data = make([]byte, bodyLen-4)
		copy(p.Data, b[4:])
	}
	return p, nil
}
