// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package icmp

import (
	"net"
	"reflect"
	"testing"

	"golang.org/x/net/internal/iana"
)

var marshalAndParseExtensionTests = []struct {
	proto int
	hdr   []byte
	obj   []byte
	exts  []Extension
}{
	// MPLS label stack with no label
	{
		proto: iana.ProtocolICMP,
		hdr: []byte{
			0x20, 0x00, 0x00, 0x00,
		},
		obj: []byte{
			0x00, 0x04, 0x01, 0x01,
		},
		exts: []Extension{
			&MPLSLabelStack{
				Class: classMPLSLabelStack,
				Type:  typeIncomingMPLSLabelStack,
			},
		},
	},
	// MPLS label stack with a single label
	{
		proto: iana.ProtocolIPv6ICMP,
		hdr: []byte{
			0x20, 0x00, 0x00, 0x00,
		},
		obj: []byte{
			0x00, 0x08, 0x01, 0x01,
			0x03, 0xe8, 0xe9, 0xff,
		},
		exts: []Extension{
			&MPLSLabelStack{
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
	},
	// MPLS label stack with multiple labels
	{
		proto: iana.ProtocolICMP,
		hdr: []byte{
			0x20, 0x00, 0x00, 0x00,
		},
		obj: []byte{
			0x00, 0x0c, 0x01, 0x01,
			0x03, 0xe8, 0xde, 0xfe,
			0x03, 0xe8, 0xe1, 0xff,
		},
		exts: []Extension{
			&MPLSLabelStack{
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
	},
	// Interface information with no attribute
	{
		proto: iana.ProtocolICMP,
		hdr: []byte{
			0x20, 0x00, 0x00, 0x00,
		},
		obj: []byte{
			0x00, 0x04, 0x02, 0x00,
		},
		exts: []Extension{
			&InterfaceInfo{
				Class: classInterfaceInfo,
			},
		},
	},
	// Interface information with ifIndex and name
	{
		proto: iana.ProtocolICMP,
		hdr: []byte{
			0x20, 0x00, 0x00, 0x00,
		},
		obj: []byte{
			0x00, 0x10, 0x02, 0x0a,
			0x00, 0x00, 0x00, 0x10,
			0x08, byte('e'), byte('n'), byte('1'),
			byte('0'), byte('1'), 0x00, 0x00,
		},
		exts: []Extension{
			&InterfaceInfo{
				Class: classInterfaceInfo,
				Type:  0x0a,
				Interface: &net.Interface{
					Index: 16,
					Name:  "en101",
				},
			},
		},
	},
	// Interface information with ifIndex, IPAddr, name and MTU
	{
		proto: iana.ProtocolIPv6ICMP,
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
		exts: []Extension{
			&InterfaceInfo{
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
	},
}

func TestMarshalAndParseExtension(t *testing.T) {
	for i, tt := range marshalAndParseExtensionTests {
		for j, ext := range tt.exts {
			var err error
			var b []byte
			switch ext := ext.(type) {
			case *MPLSLabelStack:
				b, err = ext.Marshal(tt.proto)
				if err != nil {
					t.Errorf("#%v/%v: %v", i, j, err)
					continue
				}
			case *InterfaceInfo:
				b, err = ext.Marshal(tt.proto)
				if err != nil {
					t.Errorf("#%v/%v: %v", i, j, err)
					continue
				}
			}
			if !reflect.DeepEqual(b, tt.obj) {
				t.Errorf("#%v/%v: got %#v; want %#v", i, j, b, tt.obj)
				continue
			}
		}

		for j, wire := range []struct {
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

			{append(make([]byte, 128), append(tt.hdr, tt.obj...)...), 127, 128, nil},
			{append(make([]byte, 128), append(tt.hdr, tt.obj...)...), 128, 128, nil},
			{append(make([]byte, 128), append(tt.hdr, tt.obj...)...), 129, 128, nil},

			{append(make([]byte, 512), append(tt.hdr, tt.obj...)...), 511, -1, errNoExtension},
			{append(make([]byte, 512), append(tt.hdr, tt.obj...)...), 512, 512, nil},
			{append(make([]byte, 512), append(tt.hdr, tt.obj...)...), 513, -1, errNoExtension},
		} {
			exts, l, err := parseExtensions(wire.data, wire.inlattr)
			if err != wire.err {
				t.Errorf("#%v/%v: got %v; want %v", i, j, err, wire.err)
				continue
			}
			if wire.err != nil {
				continue
			}
			if l != wire.outlattr {
				t.Errorf("#%v/%v: got %v; want %v", i, j, l, wire.outlattr)
			}
			if !reflect.DeepEqual(exts, tt.exts) {
				for j, ext := range exts {
					switch ext := ext.(type) {
					case *MPLSLabelStack:
						want := tt.exts[j].(*MPLSLabelStack)
						t.Errorf("#%v/%v: got %#v; want %#v", i, j, ext, want)
					case *InterfaceInfo:
						want := tt.exts[j].(*InterfaceInfo)
						t.Errorf("#%v/%v: got %#v; want %#v", i, j, ext, want)
					}
				}
				continue
			}
		}
	}
}

var parseInterfaceNameTests = []struct {
	b []byte
	error
}{
	{[]byte{0, 'e', 'n', '0'}, errInvalidExtension},
	{[]byte{4, 'e', 'n', '0'}, nil},
	{[]byte{7, 'e', 'n', '0', 0xff, 0xff, 0xff, 0xff}, errInvalidExtension},
	{[]byte{8, 'e', 'n', '0', 0xff, 0xff, 0xff}, errMessageTooShort},
}

func TestParseInterfaceName(t *testing.T) {
	ifi := InterfaceInfo{Interface: &net.Interface{}}
	for i, tt := range parseInterfaceNameTests {
		if _, err := ifi.parseName(tt.b); err != tt.error {
			t.Errorf("#%d: got %v; want %v", i, err, tt.error)
		}
	}
}
