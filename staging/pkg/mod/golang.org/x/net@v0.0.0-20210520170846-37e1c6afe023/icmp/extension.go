// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package icmp

import (
	"encoding/binary"

	"golang.org/x/net/ipv4"
	"golang.org/x/net/ipv6"
)

// An Extension represents an ICMP extension.
type Extension interface {
	// Len returns the length of ICMP extension.
	// The provided proto must be either the ICMPv4 or ICMPv6
	// protocol number.
	Len(proto int) int

	// Marshal returns the binary encoding of ICMP extension.
	// The provided proto must be either the ICMPv4 or ICMPv6
	// protocol number.
	Marshal(proto int) ([]byte, error)
}

const extensionVersion = 2

func validExtensionHeader(b []byte) bool {
	v := int(b[0]&0xf0) >> 4
	s := binary.BigEndian.Uint16(b[2:4])
	if s != 0 {
		s = checksum(b)
	}
	if v != extensionVersion || s != 0 {
		return false
	}
	return true
}

// parseExtensions parses b as a list of ICMP extensions.
// The length attribute l must be the length attribute field in
// received icmp messages.
//
// It will return a list of ICMP extensions and an adjusted length
// attribute that represents the length of the padded original
// datagram field. Otherwise, it returns an error.
func parseExtensions(typ Type, b []byte, l int) ([]Extension, int, error) {
	// Still a lot of non-RFC 4884 compliant implementations are
	// out there. Set the length attribute l to 128 when it looks
	// inappropriate for backwards compatibility.
	//
	// A minimal extension at least requires 8 octets; 4 octets
	// for an extension header, and 4 octets for a single object
	// header.
	//
	// See RFC 4884 for further information.
	switch typ {
	case ipv4.ICMPTypeExtendedEchoRequest, ipv6.ICMPTypeExtendedEchoRequest:
		if len(b) < 8 || !validExtensionHeader(b) {
			return nil, -1, errNoExtension
		}
		l = 0
	default:
		if 128 > l || l+8 > len(b) {
			l = 128
		}
		if l+8 > len(b) {
			return nil, -1, errNoExtension
		}
		if !validExtensionHeader(b[l:]) {
			if l == 128 {
				return nil, -1, errNoExtension
			}
			l = 128
			if !validExtensionHeader(b[l:]) {
				return nil, -1, errNoExtension
			}
		}
	}
	var exts []Extension
	for b = b[l+4:]; len(b) >= 4; {
		ol := int(binary.BigEndian.Uint16(b[:2]))
		if 4 > ol || ol > len(b) {
			break
		}
		switch b[2] {
		case classMPLSLabelStack:
			ext, err := parseMPLSLabelStack(b[:ol])
			if err != nil {
				return nil, -1, err
			}
			exts = append(exts, ext)
		case classInterfaceInfo:
			ext, err := parseInterfaceInfo(b[:ol])
			if err != nil {
				return nil, -1, err
			}
			exts = append(exts, ext)
		case classInterfaceIdent:
			ext, err := parseInterfaceIdent(b[:ol])
			if err != nil {
				return nil, -1, err
			}
			exts = append(exts, ext)
		default:
			ext := &RawExtension{Data: make([]byte, ol)}
			copy(ext.Data, b[:ol])
			exts = append(exts, ext)
		}
		b = b[ol:]
	}
	return exts, l, nil
}

func validExtensions(typ Type, exts []Extension) bool {
	switch typ {
	case ipv4.ICMPTypeDestinationUnreachable, ipv4.ICMPTypeTimeExceeded, ipv4.ICMPTypeParameterProblem,
		ipv6.ICMPTypeDestinationUnreachable, ipv6.ICMPTypeTimeExceeded:
		for i := range exts {
			switch exts[i].(type) {
			case *MPLSLabelStack, *InterfaceInfo, *RawExtension:
			default:
				return false
			}
		}
		return true
	case ipv4.ICMPTypeExtendedEchoRequest, ipv6.ICMPTypeExtendedEchoRequest:
		var n int
		for i := range exts {
			switch exts[i].(type) {
			case *InterfaceIdent:
				n++
			case *RawExtension:
			default:
				return false
			}
		}
		// Not a single InterfaceIdent object or a combo of
		// RawExtension and InterfaceIdent objects is not
		// allowed.
		if n == 1 && len(exts) > 1 {
			return false
		}
		return true
	default:
		return false
	}
}

// A RawExtension represents a raw extension.
//
// A raw extension is excluded from message processing and can be used
// to construct applications such as protocol conformance testing.
type RawExtension struct {
	Data []byte // data
}

// Len implements the Len method of Extension interface.
func (p *RawExtension) Len(proto int) int {
	if p == nil {
		return 0
	}
	return len(p.Data)
}

// Marshal implements the Marshal method of Extension interface.
func (p *RawExtension) Marshal(proto int) ([]byte, error) {
	return p.Data, nil
}
