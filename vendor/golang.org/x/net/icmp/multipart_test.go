// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package icmp_test

import (
	"fmt"
	"net"
	"reflect"
	"testing"

	"golang.org/x/net/icmp"
	"golang.org/x/net/internal/iana"
	"golang.org/x/net/ipv4"
	"golang.org/x/net/ipv6"
)

var marshalAndParseMultipartMessageForIPv4Tests = []icmp.Message{
	{
		Type: ipv4.ICMPTypeDestinationUnreachable, Code: 15,
		Body: &icmp.DstUnreach{
			Data: []byte("ERROR-INVOKING-PACKET"),
			Extensions: []icmp.Extension{
				&icmp.MPLSLabelStack{
					Class: 1,
					Type:  1,
					Labels: []icmp.MPLSLabel{
						{
							Label: 16014,
							TC:    0x4,
							S:     true,
							TTL:   255,
						},
					},
				},
				&icmp.InterfaceInfo{
					Class: 2,
					Type:  0x0f,
					Interface: &net.Interface{
						Index: 15,
						Name:  "en101",
						MTU:   8192,
					},
					Addr: &net.IPAddr{
						IP: net.IPv4(192, 168, 0, 1).To4(),
					},
				},
			},
		},
	},
	{
		Type: ipv4.ICMPTypeTimeExceeded, Code: 1,
		Body: &icmp.TimeExceeded{
			Data: []byte("ERROR-INVOKING-PACKET"),
			Extensions: []icmp.Extension{
				&icmp.InterfaceInfo{
					Class: 2,
					Type:  0x0f,
					Interface: &net.Interface{
						Index: 15,
						Name:  "en101",
						MTU:   8192,
					},
					Addr: &net.IPAddr{
						IP: net.IPv4(192, 168, 0, 1).To4(),
					},
				},
				&icmp.MPLSLabelStack{
					Class: 1,
					Type:  1,
					Labels: []icmp.MPLSLabel{
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
	},
	{
		Type: ipv4.ICMPTypeParameterProblem, Code: 2,
		Body: &icmp.ParamProb{
			Pointer: 8,
			Data:    []byte("ERROR-INVOKING-PACKET"),
			Extensions: []icmp.Extension{
				&icmp.MPLSLabelStack{
					Class: 1,
					Type:  1,
					Labels: []icmp.MPLSLabel{
						{
							Label: 16014,
							TC:    0x4,
							S:     true,
							TTL:   255,
						},
					},
				},
				&icmp.InterfaceInfo{
					Class: 2,
					Type:  0x0f,
					Interface: &net.Interface{
						Index: 15,
						Name:  "en101",
						MTU:   8192,
					},
					Addr: &net.IPAddr{
						IP: net.IPv4(192, 168, 0, 1).To4(),
					},
				},
				&icmp.InterfaceInfo{
					Class: 2,
					Type:  0x2f,
					Interface: &net.Interface{
						Index: 16,
						Name:  "en102",
						MTU:   8192,
					},
					Addr: &net.IPAddr{
						IP: net.IPv4(192, 168, 0, 2).To4(),
					},
				},
			},
		},
	},
}

func TestMarshalAndParseMultipartMessageForIPv4(t *testing.T) {
	for i, tt := range marshalAndParseMultipartMessageForIPv4Tests {
		b, err := tt.Marshal(nil)
		if err != nil {
			t.Fatal(err)
		}
		if b[5] != 32 {
			t.Errorf("#%v: got %v; want 32", i, b[5])
		}
		m, err := icmp.ParseMessage(iana.ProtocolICMP, b)
		if err != nil {
			t.Fatal(err)
		}
		if m.Type != tt.Type || m.Code != tt.Code {
			t.Errorf("#%v: got %v; want %v", i, m, &tt)
		}
		switch m.Type {
		case ipv4.ICMPTypeDestinationUnreachable:
			got, want := m.Body.(*icmp.DstUnreach), tt.Body.(*icmp.DstUnreach)
			if !reflect.DeepEqual(got.Extensions, want.Extensions) {
				t.Error(dumpExtensions(i, got.Extensions, want.Extensions))
			}
			if len(got.Data) != 128 {
				t.Errorf("#%v: got %v; want 128", i, len(got.Data))
			}
		case ipv4.ICMPTypeTimeExceeded:
			got, want := m.Body.(*icmp.TimeExceeded), tt.Body.(*icmp.TimeExceeded)
			if !reflect.DeepEqual(got.Extensions, want.Extensions) {
				t.Error(dumpExtensions(i, got.Extensions, want.Extensions))
			}
			if len(got.Data) != 128 {
				t.Errorf("#%v: got %v; want 128", i, len(got.Data))
			}
		case ipv4.ICMPTypeParameterProblem:
			got, want := m.Body.(*icmp.ParamProb), tt.Body.(*icmp.ParamProb)
			if !reflect.DeepEqual(got.Extensions, want.Extensions) {
				t.Error(dumpExtensions(i, got.Extensions, want.Extensions))
			}
			if len(got.Data) != 128 {
				t.Errorf("#%v: got %v; want 128", i, len(got.Data))
			}
		}
	}
}

var marshalAndParseMultipartMessageForIPv6Tests = []icmp.Message{
	{
		Type: ipv6.ICMPTypeDestinationUnreachable, Code: 6,
		Body: &icmp.DstUnreach{
			Data: []byte("ERROR-INVOKING-PACKET"),
			Extensions: []icmp.Extension{
				&icmp.MPLSLabelStack{
					Class: 1,
					Type:  1,
					Labels: []icmp.MPLSLabel{
						{
							Label: 16014,
							TC:    0x4,
							S:     true,
							TTL:   255,
						},
					},
				},
				&icmp.InterfaceInfo{
					Class: 2,
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
	},
	{
		Type: ipv6.ICMPTypeTimeExceeded, Code: 1,
		Body: &icmp.TimeExceeded{
			Data: []byte("ERROR-INVOKING-PACKET"),
			Extensions: []icmp.Extension{
				&icmp.InterfaceInfo{
					Class: 2,
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
				&icmp.MPLSLabelStack{
					Class: 1,
					Type:  1,
					Labels: []icmp.MPLSLabel{
						{
							Label: 16014,
							TC:    0x4,
							S:     true,
							TTL:   255,
						},
					},
				},
				&icmp.InterfaceInfo{
					Class: 2,
					Type:  0x2f,
					Interface: &net.Interface{
						Index: 16,
						Name:  "en102",
						MTU:   8192,
					},
					Addr: &net.IPAddr{
						IP:   net.ParseIP("fe80::1"),
						Zone: "en102",
					},
				},
			},
		},
	},
}

func TestMarshalAndParseMultipartMessageForIPv6(t *testing.T) {
	pshicmp := icmp.IPv6PseudoHeader(net.ParseIP("fe80::1"), net.ParseIP("ff02::1"))
	for i, tt := range marshalAndParseMultipartMessageForIPv6Tests {
		for _, psh := range [][]byte{pshicmp, nil} {
			b, err := tt.Marshal(psh)
			if err != nil {
				t.Fatal(err)
			}
			if b[4] != 16 {
				t.Errorf("#%v: got %v; want 16", i, b[4])
			}
			m, err := icmp.ParseMessage(iana.ProtocolIPv6ICMP, b)
			if err != nil {
				t.Fatal(err)
			}
			if m.Type != tt.Type || m.Code != tt.Code {
				t.Errorf("#%v: got %v; want %v", i, m, &tt)
			}
			switch m.Type {
			case ipv6.ICMPTypeDestinationUnreachable:
				got, want := m.Body.(*icmp.DstUnreach), tt.Body.(*icmp.DstUnreach)
				if !reflect.DeepEqual(got.Extensions, want.Extensions) {
					t.Error(dumpExtensions(i, got.Extensions, want.Extensions))
				}
				if len(got.Data) != 128 {
					t.Errorf("#%v: got %v; want 128", i, len(got.Data))
				}
			case ipv6.ICMPTypeTimeExceeded:
				got, want := m.Body.(*icmp.TimeExceeded), tt.Body.(*icmp.TimeExceeded)
				if !reflect.DeepEqual(got.Extensions, want.Extensions) {
					t.Error(dumpExtensions(i, got.Extensions, want.Extensions))
				}
				if len(got.Data) != 128 {
					t.Errorf("#%v: got %v; want 128", i, len(got.Data))
				}
			}
		}
	}
}

func dumpExtensions(i int, gotExts, wantExts []icmp.Extension) string {
	var s string
	for j, got := range gotExts {
		switch got := got.(type) {
		case *icmp.MPLSLabelStack:
			want := wantExts[j].(*icmp.MPLSLabelStack)
			if !reflect.DeepEqual(got, want) {
				s += fmt.Sprintf("#%v/%v: got %#v; want %#v\n", i, j, got, want)
			}
		case *icmp.InterfaceInfo:
			want := wantExts[j].(*icmp.InterfaceInfo)
			if !reflect.DeepEqual(got, want) {
				s += fmt.Sprintf("#%v/%v: got %#v, %#v, %#v; want %#v, %#v, %#v\n", i, j, got, got.Interface, got.Addr, want, want.Interface, want.Addr)
			}
		}
	}
	return s[:len(s)-1]
}
