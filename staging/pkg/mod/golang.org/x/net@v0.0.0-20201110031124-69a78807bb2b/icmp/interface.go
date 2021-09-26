// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package icmp

import (
	"encoding/binary"
	"net"
	"strings"

	"golang.org/x/net/internal/iana"
)

const (
	classInterfaceInfo = 2
)

const (
	attrMTU = 1 << iota
	attrName
	attrIPAddr
	attrIfIndex
)

// An InterfaceInfo represents interface and next-hop identification.
type InterfaceInfo struct {
	Class     int // extension object class number
	Type      int // extension object sub-type
	Interface *net.Interface
	Addr      *net.IPAddr
}

func (ifi *InterfaceInfo) nameLen() int {
	if len(ifi.Interface.Name) > 63 {
		return 64
	}
	l := 1 + len(ifi.Interface.Name)
	return (l + 3) &^ 3
}

func (ifi *InterfaceInfo) attrsAndLen(proto int) (attrs, l int) {
	l = 4
	if ifi.Interface != nil && ifi.Interface.Index > 0 {
		attrs |= attrIfIndex
		l += 4
		if len(ifi.Interface.Name) > 0 {
			attrs |= attrName
			l += ifi.nameLen()
		}
		if ifi.Interface.MTU > 0 {
			attrs |= attrMTU
			l += 4
		}
	}
	if ifi.Addr != nil {
		switch proto {
		case iana.ProtocolICMP:
			if ifi.Addr.IP.To4() != nil {
				attrs |= attrIPAddr
				l += 4 + net.IPv4len
			}
		case iana.ProtocolIPv6ICMP:
			if ifi.Addr.IP.To16() != nil && ifi.Addr.IP.To4() == nil {
				attrs |= attrIPAddr
				l += 4 + net.IPv6len
			}
		}
	}
	return
}

// Len implements the Len method of Extension interface.
func (ifi *InterfaceInfo) Len(proto int) int {
	_, l := ifi.attrsAndLen(proto)
	return l
}

// Marshal implements the Marshal method of Extension interface.
func (ifi *InterfaceInfo) Marshal(proto int) ([]byte, error) {
	attrs, l := ifi.attrsAndLen(proto)
	b := make([]byte, l)
	if err := ifi.marshal(proto, b, attrs, l); err != nil {
		return nil, err
	}
	return b, nil
}

func (ifi *InterfaceInfo) marshal(proto int, b []byte, attrs, l int) error {
	binary.BigEndian.PutUint16(b[:2], uint16(l))
	b[2], b[3] = classInterfaceInfo, byte(ifi.Type)
	for b = b[4:]; len(b) > 0 && attrs != 0; {
		switch {
		case attrs&attrIfIndex != 0:
			b = ifi.marshalIfIndex(proto, b)
			attrs &^= attrIfIndex
		case attrs&attrIPAddr != 0:
			b = ifi.marshalIPAddr(proto, b)
			attrs &^= attrIPAddr
		case attrs&attrName != 0:
			b = ifi.marshalName(proto, b)
			attrs &^= attrName
		case attrs&attrMTU != 0:
			b = ifi.marshalMTU(proto, b)
			attrs &^= attrMTU
		}
	}
	return nil
}

func (ifi *InterfaceInfo) marshalIfIndex(proto int, b []byte) []byte {
	binary.BigEndian.PutUint32(b[:4], uint32(ifi.Interface.Index))
	return b[4:]
}

func (ifi *InterfaceInfo) parseIfIndex(b []byte) ([]byte, error) {
	if len(b) < 4 {
		return nil, errMessageTooShort
	}
	ifi.Interface.Index = int(binary.BigEndian.Uint32(b[:4]))
	return b[4:], nil
}

func (ifi *InterfaceInfo) marshalIPAddr(proto int, b []byte) []byte {
	switch proto {
	case iana.ProtocolICMP:
		binary.BigEndian.PutUint16(b[:2], uint16(iana.AddrFamilyIPv4))
		copy(b[4:4+net.IPv4len], ifi.Addr.IP.To4())
		b = b[4+net.IPv4len:]
	case iana.ProtocolIPv6ICMP:
		binary.BigEndian.PutUint16(b[:2], uint16(iana.AddrFamilyIPv6))
		copy(b[4:4+net.IPv6len], ifi.Addr.IP.To16())
		b = b[4+net.IPv6len:]
	}
	return b
}

func (ifi *InterfaceInfo) parseIPAddr(b []byte) ([]byte, error) {
	if len(b) < 4 {
		return nil, errMessageTooShort
	}
	afi := int(binary.BigEndian.Uint16(b[:2]))
	b = b[4:]
	switch afi {
	case iana.AddrFamilyIPv4:
		if len(b) < net.IPv4len {
			return nil, errMessageTooShort
		}
		ifi.Addr.IP = make(net.IP, net.IPv4len)
		copy(ifi.Addr.IP, b[:net.IPv4len])
		b = b[net.IPv4len:]
	case iana.AddrFamilyIPv6:
		if len(b) < net.IPv6len {
			return nil, errMessageTooShort
		}
		ifi.Addr.IP = make(net.IP, net.IPv6len)
		copy(ifi.Addr.IP, b[:net.IPv6len])
		b = b[net.IPv6len:]
	}
	return b, nil
}

func (ifi *InterfaceInfo) marshalName(proto int, b []byte) []byte {
	l := byte(ifi.nameLen())
	b[0] = l
	copy(b[1:], []byte(ifi.Interface.Name))
	return b[l:]
}

func (ifi *InterfaceInfo) parseName(b []byte) ([]byte, error) {
	if 4 > len(b) || len(b) < int(b[0]) {
		return nil, errMessageTooShort
	}
	l := int(b[0])
	if l%4 != 0 || 4 > l || l > 64 {
		return nil, errInvalidExtension
	}
	var name [63]byte
	copy(name[:], b[1:l])
	ifi.Interface.Name = strings.Trim(string(name[:]), "\000")
	return b[l:], nil
}

func (ifi *InterfaceInfo) marshalMTU(proto int, b []byte) []byte {
	binary.BigEndian.PutUint32(b[:4], uint32(ifi.Interface.MTU))
	return b[4:]
}

func (ifi *InterfaceInfo) parseMTU(b []byte) ([]byte, error) {
	if len(b) < 4 {
		return nil, errMessageTooShort
	}
	ifi.Interface.MTU = int(binary.BigEndian.Uint32(b[:4]))
	return b[4:], nil
}

func parseInterfaceInfo(b []byte) (Extension, error) {
	ifi := &InterfaceInfo{
		Class: int(b[2]),
		Type:  int(b[3]),
	}
	if ifi.Type&(attrIfIndex|attrName|attrMTU) != 0 {
		ifi.Interface = &net.Interface{}
	}
	if ifi.Type&attrIPAddr != 0 {
		ifi.Addr = &net.IPAddr{}
	}
	attrs := ifi.Type & (attrIfIndex | attrIPAddr | attrName | attrMTU)
	for b = b[4:]; len(b) > 0 && attrs != 0; {
		var err error
		switch {
		case attrs&attrIfIndex != 0:
			b, err = ifi.parseIfIndex(b)
			attrs &^= attrIfIndex
		case attrs&attrIPAddr != 0:
			b, err = ifi.parseIPAddr(b)
			attrs &^= attrIPAddr
		case attrs&attrName != 0:
			b, err = ifi.parseName(b)
			attrs &^= attrName
		case attrs&attrMTU != 0:
			b, err = ifi.parseMTU(b)
			attrs &^= attrMTU
		}
		if err != nil {
			return nil, err
		}
	}
	if ifi.Interface != nil && ifi.Interface.Name != "" && ifi.Addr != nil && ifi.Addr.IP.To16() != nil && ifi.Addr.IP.To4() == nil {
		ifi.Addr.Zone = ifi.Interface.Name
	}
	return ifi, nil
}

const (
	classInterfaceIdent    = 3
	typeInterfaceByName    = 1
	typeInterfaceByIndex   = 2
	typeInterfaceByAddress = 3
)

// An InterfaceIdent represents interface identification.
type InterfaceIdent struct {
	Class int    // extension object class number
	Type  int    // extension object sub-type
	Name  string // interface name
	Index int    // interface index
	AFI   int    // address family identifier; see address family numbers in IANA registry
	Addr  []byte // address
}

// Len implements the Len method of Extension interface.
func (ifi *InterfaceIdent) Len(_ int) int {
	switch ifi.Type {
	case typeInterfaceByName:
		l := len(ifi.Name)
		if l > 255 {
			l = 255
		}
		return 4 + (l+3)&^3
	case typeInterfaceByIndex:
		return 4 + 4
	case typeInterfaceByAddress:
		return 4 + 4 + (len(ifi.Addr)+3)&^3
	default:
		return 4
	}
}

// Marshal implements the Marshal method of Extension interface.
func (ifi *InterfaceIdent) Marshal(proto int) ([]byte, error) {
	b := make([]byte, ifi.Len(proto))
	if err := ifi.marshal(proto, b); err != nil {
		return nil, err
	}
	return b, nil
}

func (ifi *InterfaceIdent) marshal(proto int, b []byte) error {
	l := ifi.Len(proto)
	binary.BigEndian.PutUint16(b[:2], uint16(l))
	b[2], b[3] = classInterfaceIdent, byte(ifi.Type)
	switch ifi.Type {
	case typeInterfaceByName:
		copy(b[4:], ifi.Name)
	case typeInterfaceByIndex:
		binary.BigEndian.PutUint32(b[4:4+4], uint32(ifi.Index))
	case typeInterfaceByAddress:
		binary.BigEndian.PutUint16(b[4:4+2], uint16(ifi.AFI))
		b[4+2] = byte(len(ifi.Addr))
		copy(b[4+4:], ifi.Addr)
	}
	return nil
}

func parseInterfaceIdent(b []byte) (Extension, error) {
	ifi := &InterfaceIdent{
		Class: int(b[2]),
		Type:  int(b[3]),
	}
	switch ifi.Type {
	case typeInterfaceByName:
		ifi.Name = strings.Trim(string(b[4:]), "\x00")
	case typeInterfaceByIndex:
		if len(b[4:]) < 4 {
			return nil, errInvalidExtension
		}
		ifi.Index = int(binary.BigEndian.Uint32(b[4 : 4+4]))
	case typeInterfaceByAddress:
		if len(b[4:]) < 4 {
			return nil, errInvalidExtension
		}
		ifi.AFI = int(binary.BigEndian.Uint16(b[4 : 4+2]))
		l := int(b[4+2])
		if len(b[4+4:]) < l {
			return nil, errInvalidExtension
		}
		ifi.Addr = make([]byte, l)
		copy(ifi.Addr, b[4+4:])
	}
	return ifi, nil
}
