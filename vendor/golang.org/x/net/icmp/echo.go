// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package icmp

import "encoding/binary"

// An Echo represents an ICMP echo request or reply message body.
type Echo struct {
	ID   int    // identifier
	Seq  int    // sequence number
	Data []byte // data
}

// Len implements the Len method of MessageBody interface.
func (p *Echo) Len(proto int) int {
	if p == nil {
		return 0
	}
	return 4 + len(p.Data)
}

// Marshal implements the Marshal method of MessageBody interface.
func (p *Echo) Marshal(proto int) ([]byte, error) {
	b := make([]byte, 4+len(p.Data))
	binary.BigEndian.PutUint16(b[:2], uint16(p.ID))
	binary.BigEndian.PutUint16(b[2:4], uint16(p.Seq))
	copy(b[4:], p.Data)
	return b, nil
}

// parseEcho parses b as an ICMP echo request or reply message body.
func parseEcho(proto int, b []byte) (MessageBody, error) {
	bodyLen := len(b)
	if bodyLen < 4 {
		return nil, errMessageTooShort
	}
	p := &Echo{ID: int(binary.BigEndian.Uint16(b[:2])), Seq: int(binary.BigEndian.Uint16(b[2:4]))}
	if bodyLen > 4 {
		p.Data = make([]byte, bodyLen-4)
		copy(p.Data, b[4:])
	}
	return p, nil
}
