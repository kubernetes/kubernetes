// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package icmp

import (
	"fmt"
	"net"
	"reflect"
	"testing"

	"golang.org/x/net/internal/iana"
	"golang.org/x/net/ipv4"
	"golang.org/x/net/ipv6"
)

func TestMarshalAndParseExtension(t *testing.T) {
	fn := func(t *testing.T, proto int, typ Type, hdr, obj []byte, te Extension) error {
		b, err := te.Marshal(proto)
		if err != nil {
			return err
		}
		if !reflect.DeepEqual(b, obj) {
			return fmt.Errorf("got %#v; want %#v", b, obj)
		}
		switch typ {
		case ipv4.ICMPTypeExtendedEchoRequest, ipv6.ICMPTypeExtendedEchoRequest:
			exts, l, err := parseExtensions(typ, append(hdr, obj...), 0)
			if err != nil {
				return err
			}
			if l != 0 {
				return fmt.Errorf("got %d; want 0", l)
			}
			if !reflect.DeepEqual(exts, []Extension{te}) {
				return fmt.Errorf("got %#v; want %#v", exts[0], te)
			}
		default:
			for i, wire := range []struct {
				data     []byte // original datagram
				inlattr  int    // length of padded original datagram, a hint
				outlattr int    // length of padded original datagram, a want
				err      error
			}{
				{nil, 0, -1, errNoExtension},
				{make([]byte, 127), 128, -1, errNoExtension},

				{make([]byte, 128), 127, -1, errNoExtension},
				{make([]byte, 128), 128, -1, errNoExtension},
				{make([]byte, 128), 129, -1, errNoExtension},

				{append(make([]byte, 128), append(hdr, obj...)...), 127, 128, nil},
				{append(make([]byte, 128), append(hdr, obj...)...), 128, 128, nil},
				{append(make([]byte, 128), append(hdr, obj...)...), 129, 128, nil},

				{append(make([]byte, 512), append(hdr, obj...)...), 511, -1, errNoExtension},
				{append(make([]byte, 512), append(hdr, obj...)...), 512, 512, nil},
				{append(make([]byte, 512), append(hdr, obj...)...), 513, -1, errNoExtension},
			} {
				exts, l, err := parseExtensions(typ, wire.data, wire.inlattr)
				if err != wire.err {
					return fmt.Errorf("#%d: got %v; want %v", i, err, wire.err)
				}
				if wire.err != nil {
					continue
				}
				if l != wire.outlattr {
					return fmt.Errorf("#%d: got %d; want %d", i, l, wire.outlattr)
				}
				if !reflect.DeepEqual(exts, []Extension{te}) {
					return fmt.Errorf("#%d: got %#v; want %#v", i, exts[0], te)
				}
			}
		}
		return nil
	}

	t.Run("MPLSLabelStack", func(t *testing.T) {
		for _, et := range []struct {
			proto int
			typ   Type
			hdr   []byte
			obj   []byte
			ext   Extension
		}{
			// MPLS label stack with no label
			{
				proto: iana.ProtocolICMP,
				typ:   ipv4.ICMPTypeDestinationUnreachable,
				hdr: []byte{
					0x20, 0x00, 0x00, 0x00,
				},
				obj: []byte{
					0x00, 0x04, 0x01, 0x01,
				},
				ext: &MPLSLabelStack{
					Class: classMPLSLabelStack,
					Type:  typeIncomingMPLSLabelStack,
				},
			},
			// MPLS label stack with a single label
			{
				proto: iana.ProtocolIPv6ICMP,
				typ:   ipv6.ICMPTypeDestinationUnreachable,
				hdr: []byte{
					0x20, 0x00, 0x00, 0x00,
				},
				obj: []byte{
					0x00, 0x08, 0x01, 0x01,
					0x03, 0xe8, 0xe9, 0xff,
				},
				ext: &MPLSLabelStack{
					Class: classMPLSLabelStack,
					Type:  typeIncomingMPLSLabelStack,
					Labels: []MPLSLabel{
						{
							Label: 16014,
							TC:    0x4,
							S:     true,
							TTL:   255,
						},
					},
				},
			},
			// MPLS label stack with multiple labels
			{
				proto: iana.ProtocolICMP,
				typ:   ipv4.ICMPTypeDestinationUnreachable,
				hdr: []byte{
					0x20, 0x00, 0x00, 0x00,
				},
				obj: []byte{
					0x00, 0x0c, 0x01, 0x01,
					0x03, 0xe8, 0xde, 0xfe,
					0x03, 0xe8, 0xe1, 0xff,
				},
				ext: &MPLSLabelStack{
					Class: classMPLSLabelStack,
					Type:  typeIncomingMPLSLabelStack,
					Labels: []MPLSLabel{
						{
							Label: 16013,
							TC:    0x7,
							S:     false,
							TTL:   254,
						},
						{
							Label: 16014,
							TC:    0,
							S:     true,
							TTL:   255,
						},
					},
				},
			},
		} {
			if err := fn(t, et.proto, et.typ, et.hdr, et.obj, et.ext); err != nil {
				t.Error(err)
			}
		}
	})
	t.Run("InterfaceInfo", func(t *testing.T) {
		for _, et := range []struct {
			proto int
			typ   Type
			hdr   []byte
			obj   []byte
			ext   Extension
		}{
			// Interface information with no attribute
			{
				proto: iana.ProtocolICMP,
				typ:   ipv4.ICMPTypeDestinationUnreachable,
				hdr: []byte{
					0x20, 0x00, 0x00, 0x00,
				},
				obj: []byte{
					0x00, 0x04, 0x02, 0x00,
				},
				ext: &InterfaceInfo{
					Class: classInterfaceInfo,
				},
			},
			// Interface information with ifIndex and name
			{
				proto: iana.ProtocolICMP,
				typ:   ipv4.ICMPTypeDestinationUnreachable,
				hdr: []byte{
					0x20, 0x00, 0x00, 0x00,
				},
				obj: []byte{
					0x00, 0x10, 0x02, 0x0a,
					0x00, 0x00, 0x00, 0x10,
					0x08, byte('e'), byte('n'), byte('1'),
					byte('0'), byte('1'), 0x00, 0x00,
				},
				ext: &InterfaceInfo{
					Class: classInterfaceInfo,
					Type:  0x0a,
					Interface: &net.Interface{
						Index: 16,
						Name:  "en101",
					},
				},
			},
			// Interface information with ifIndex, IPAddr, name and MTU
			{
				proto: iana.ProtocolIPv6ICMP,
				typ:   ipv6.ICMPTypeDestinationUnreachable,
				hdr: []byte{
					0x20, 0x00, 0x00, 0x00,
				},
				obj: []byte{
					0x00, 0x28, 0x02, 0x0f,
					0x00, 0x00, 0x00, 0x0f,
					0x00, 0x02, 0x00, 0x00,
					0xfe, 0x80, 0x00, 0x00,
					0x00, 0x00, 0x00, 0x00,
					0x00, 0x00, 0x00, 0x00,
					0x00, 0x00, 0x00, 0x01,
					0x08, byte('e'), byte('n'), byte('1'),
					byte('0'), byte('1'), 0x00, 0x00,
					0x00, 0x00, 0x20, 0x00,
				},
				ext: &InterfaceInfo{
					Class: classInterfaceInfo,
					Type:  0x0f,
					Interface: &net.Interface{
						Index: 15,
						Name:  "en101",
						MTU:   8192,
					},
					Addr: &net.IPAddr{
						IP:   net.ParseIP("fe80::1"),
						Zone: "en101",
					},
				},
			},
		} {
			if err := fn(t, et.proto, et.typ, et.hdr, et.obj, et.ext); err != nil {
				t.Error(err)
			}
		}
	})
	t.Run("InterfaceIdent", func(t *testing.T) {
		for _, et := range []struct {
			proto int
			typ   Type
			hdr   []byte
			obj   []byte
			ext   Extension
		}{
			// Interface identification by name
			{
				proto: iana.ProtocolICMP,
				typ:   ipv4.ICMPTypeExtendedEchoRequest,
				hdr: []byte{
					0x20, 0x00, 0x00, 0x00,
				},
				obj: []byte{
					0x00, 0x0c, 0x03, 0x01,
					byte('e'), byte('n'), byte('1'), byte('0'),
					byte('1'), 0x00, 0x00, 0x00,
				},
				ext: &InterfaceIdent{
					Class: classInterfaceIdent,
					Type:  typeInterfaceByName,
					Name:  "en101",
				},
			},
			// Interface identification by index
			{
				proto: iana.ProtocolIPv6ICMP,
				typ:   ipv6.ICMPTypeExtendedEchoRequest,
				hdr: []byte{
					0x20, 0x00, 0x00, 0x00,
				},
				obj: []byte{
					0x00, 0x08, 0x03, 0x02,
					0x00, 0x00, 0x03, 0x8f,
				},
				ext: &InterfaceIdent{
					Class: classInterfaceIdent,
					Type:  typeInterfaceByIndex,
					Index: 911,
				},
			},
			// Interface identification by address
			{
				proto: iana.ProtocolICMP,
				typ:   ipv4.ICMPTypeExtendedEchoRequest,
				hdr: []byte{
					0x20, 0x00, 0x00, 0x00,
				},
				obj: []byte{
					0x00, 0x10, 0x03, 0x03,
					byte(iana.AddrFamily48bitMAC >> 8), byte(iana.AddrFamily48bitMAC & 0x0f), 0x06, 0x00,
					0x01, 0x23, 0x45, 0x67,
					0x89, 0xab, 0x00, 0x00,
				},
				ext: &InterfaceIdent{
					Class: classInterfaceIdent,
					Type:  typeInterfaceByAddress,
					AFI:   iana.AddrFamily48bitMAC,
					Addr:  []byte{0x01, 0x23, 0x45, 0x67, 0x89, 0xab},
				},
			},
		} {
			if err := fn(t, et.proto, et.typ, et.hdr, et.obj, et.ext); err != nil {
				t.Error(err)
			}
		}
	})
}

func TestParseInterfaceName(t *testing.T) {
	ifi := InterfaceInfo{Interface: &net.Interface{}}
	for i, tt := range []struct {
		b []byte
		error
	}{
		{[]byte{0, 'e', 'n', '0'}, errInvalidExtension},
		{[]byte{4, 'e', 'n', '0'}, nil},
		{[]byte{7, 'e', 'n', '0', 0xff, 0xff, 0xff, 0xff}, errInvalidExtension},
		{[]byte{8, 'e', 'n', '0', 0xff, 0xff, 0xff}, errMessageTooShort},
	} {
		if _, err := ifi.parseName(tt.b); err != tt.error {
			t.Errorf("#%d: got %v; want %v", i, err, tt.error)
		}
	}
}
