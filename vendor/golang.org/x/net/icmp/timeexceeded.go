// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package icmp

// A TimeExceeded represents an ICMP time exceeded message body.
type TimeExceeded struct {
	Data       []byte      // data, known as original datagram field
	Extensions []Extension // extensions
}

// Len implements the Len method of MessageBody interface.
func (p *TimeExceeded) Len(proto int) int {
	if p == nil {
		return 0
	}
	l, _ := multipartMessageBodyDataLen(proto, p.Data, p.Extensions)
	return l
}

// Marshal implements the Marshal method of MessageBody interface.
func (p *TimeExceeded) Marshal(proto int) ([]byte, error) {
	return marshalMultipartMessageBody(proto, p.Data, p.Extensions)
}

// parseTimeExceeded parses b as an ICMP time exceeded message body.
func parseTimeExceeded(proto int, b []byte) (MessageBody, error) {
	if len(b) < 4 {
		return nil, errMessageTooShort
	}
	p := &TimeExceeded{}
	var err error
	p.Data, p.Extensions, err = parseMultipartMessageBody(proto, b)
	if err != nil {
		return nil, err
	}
	return p, nil
}
