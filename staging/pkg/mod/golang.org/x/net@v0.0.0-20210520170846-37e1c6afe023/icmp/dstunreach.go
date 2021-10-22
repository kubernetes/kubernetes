// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package icmp

import (
	"golang.org/x/net/internal/iana"
	"golang.org/x/net/ipv4"
	"golang.org/x/net/ipv6"
)

// A DstUnreach represents an ICMP destination unreachable message
// body.
type DstUnreach struct {
	Data       []byte      // data, known as original datagram field
	Extensions []Extension // extensions
}

// Len implements the Len method of MessageBody interface.
func (p *DstUnreach) Len(proto int) int {
	if p == nil {
		return 0
	}
	l, _ := multipartMessageBodyDataLen(proto, true, p.Data, p.Extensions)
	return l
}

// Marshal implements the Marshal method of MessageBody interface.
func (p *DstUnreach) Marshal(proto int) ([]byte, error) {
	var typ Type
	switch proto {
	case iana.ProtocolICMP:
		typ = ipv4.ICMPTypeDestinationUnreachable
	case iana.ProtocolIPv6ICMP:
		typ = ipv6.ICMPTypeDestinationUnreachable
	default:
		return nil, errInvalidProtocol
	}
	if !validExtensions(typ, p.Extensions) {
		return nil, errInvalidExtension
	}
	return marshalMultipartMessageBody(proto, true, p.Data, p.Extensions)
}

// parseDstUnreach parses b as an ICMP destination unreachable message
// body.
func parseDstUnreach(proto int, typ Type, b []byte) (MessageBody, error) {
	if len(b) < 4 {
		return nil, errMessageTooShort
	}
	p := &DstUnreach{}
	var err error
	p.Data, p.Extensions, err = parseMultipartMessageBody(proto, typ, b)
	if err != nil {
		return nil, err
	}
	return p, nil
}
