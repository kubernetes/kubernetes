// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package icmp

// A MessageBody represents an ICMP message body.
type MessageBody interface {
	// Len returns the length of ICMP message body.
	// The provided proto must be either the ICMPv4 or ICMPv6
	// protocol number.
	Len(proto int) int

	// Marshal returns the binary encoding of ICMP message body.
	// The provided proto must be either the ICMPv4 or ICMPv6
	// protocol number.
	Marshal(proto int) ([]byte, error)
}

// A RawBody represents a raw message body.
//
// A raw message body is excluded from message processing and can be
// used to construct applications such as protocol conformance
// testing.
type RawBody struct {
	Data []byte // data
}

// Len implements the Len method of MessageBody interface.
func (p *RawBody) Len(proto int) int {
	if p == nil {
		return 0
	}
	return len(p.Data)
}

// Marshal implements the Marshal method of MessageBody interface.
func (p *RawBody) Marshal(proto int) ([]byte, error) {
	return p.Data, nil
}

// parseRawBody parses b as an ICMP message body.
func parseRawBody(proto int, b []byte) (MessageBody, error) {
	p := &RawBody{Data: make([]byte, len(b))}
	copy(p.Data, b)
	return p, nil
}

// A DefaultMessageBody represents the default message body.
//
// Deprecated: Use RawBody instead.
type DefaultMessageBody = RawBody
