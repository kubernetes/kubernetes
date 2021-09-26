// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package icmp_test

import (
	"errors"
	"fmt"
	"net"
	"reflect"
	"testing"

	"golang.org/x/net/icmp"
	"golang.org/x/net/internal/iana"
	"golang.org/x/net/ipv4"
	"golang.org/x/net/ipv6"
)

func TestMarshalAndParseMultipartMessage(t *testing.T) {
	fn := func(t *testing.T, proto int, tm icmp.Message) error {
		b, err := tm.Marshal(nil)
		if err != nil {
			return err
		}
		switch tm.Type {
		case ipv4.ICMPTypeExtendedEchoRequest, ipv6.ICMPTypeExtendedEchoRequest:
		default:
			switch proto {
			case iana.ProtocolICMP:
				if b[5] != 32 {
					return fmt.Errorf("got %d; want 32", b[5])
				}
			case iana.ProtocolIPv6ICMP:
				if b[4] != 16 {
					return fmt.Errorf("got %d; want 16", b[4])
				}
			default:
				return fmt.Errorf("unknown protocol: %d", proto)
			}
		}
		m, err := icmp.ParseMessage(proto, b)
		if err != nil {
			return err
		}
		if m.Type != tm.Type || m.Code != tm.Code {
			return fmt.Errorf("got %v; want %v", m, &tm)
		}
		switch m.Type {
		case ipv4.ICMPTypeExtendedEchoRequest, ipv6.ICMPTypeExtendedEchoRequest:
			got, want := m.Body.(*icmp.ExtendedEchoRequest), tm.Body.(*icmp.ExtendedEchoRequest)
			if !reflect.DeepEqual(got.Extensions, want.Extensions) {
				return errors.New(dumpExtensions(got.Extensions, want.Extensions))
			}
		case ipv4.ICMPTypeDestinationUnreachable:
			got, want := m.Body.(*icmp.DstUnreach), tm.Body.(*icmp.DstUnreach)
			if !reflect.DeepEqual(got.Extensions, want.Extensions) {
				return errors.New(dumpExtensions(got.Extensions, want.Extensions))
			}
			if len(got.Data) != 128 {
				return fmt.Errorf("got %d; want 128", len(got.Data))
			}
		case ipv4.ICMPTypeTimeExceeded:
			got, want := m.Body.(*icmp.TimeExceeded), tm.Body.(*icmp.TimeExceeded)
			if !reflect.DeepEqual(got.Extensions, want.Extensions) {
				return errors.New(dumpExtensions(got.Extensions, want.Extensions))
			}
			if len(got.Data) != 128 {
				return fmt.Errorf("got %d; want 128", len(got.Data))
			}
		case ipv4.ICMPTypeParameterProblem:
			got, want := m.Body.(*icmp.ParamProb), tm.Body.(*icmp.ParamProb)
			if !reflect.DeepEqual(got.Extensions, want.Extensions) {
				return errors.New(dumpExtensions(got.Extensions, want.Extensions))
			}
			if len(got.Data) != 128 {
				return fmt.Errorf("got %d; want 128", len(got.Data))
			}
		case ipv6.ICMPTypeDestinationUnreachable:
			got, want := m.Body.(*icmp.DstUnreach), tm.Body.(*icmp.DstUnreach)
			if !reflect.DeepEqual(got.Extensions, want.Extensions) {
				return errors.New(dumpExtensions(got.Extensions, want.Extensions))
			}
			if len(got.Data) != 128 {
				return fmt.Errorf("got %d; want 128", len(got.Data))
			}
		case ipv6.ICMPTypeTimeExceeded:
			got, want := m.Body.(*icmp.TimeExceeded), tm.Body.(*icmp.TimeExceeded)
			if !reflect.DeepEqual(got.Extensions, want.Extensions) {
				return errors.New(dumpExtensions(got.Extensions, want.Extensions))
			}
			if len(got.Data) != 128 {
				return fmt.Errorf("got %d; want 128", len(got.Data))
			}
		default:
			return fmt.Errorf("unknown message type: %v", m.Type)
		}
		return nil
	}

	t.Run("IPv4", func(t *testing.T) {
		for i, tm := range []icmp.Message{
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
			{
				Type: ipv4.ICMPTypeExtendedEchoRequest, Code: 0,
				Body: &icmp.ExtendedEchoRequest{
					ID: 1, Seq: 2, Local: true,
					Extensions: []icmp.Extension{
						&icmp.InterfaceIdent{
							Class: 3,
							Type:  1,
							Name:  "en101",
						},
					},
				},
			},
			{
				Type: ipv4.ICMPTypeExtendedEchoRequest, Code: 0,
				Body: &icmp.ExtendedEchoRequest{
					ID: 1, Seq: 2, Local: true,
					Extensions: []icmp.Extension{
						&icmp.InterfaceIdent{
							Class: 3,
							Type:  2,
							Index: 911,
						},
					},
				},
			},
			{
				Type: ipv4.ICMPTypeExtendedEchoRequest, Code: 0,
				Body: &icmp.ExtendedEchoRequest{
					ID: 1, Seq: 2,
					Extensions: []icmp.Extension{
						&icmp.InterfaceIdent{
							Class: 3,
							Type:  3,
							AFI:   iana.AddrFamily48bitMAC,
							Addr:  []byte{0x01, 0x23, 0x45, 0x67, 0x89, 0xab},
						},
					},
				},
			},
		} {
			if err := fn(t, iana.ProtocolICMP, tm); err != nil {
				t.Errorf("#%d: %v", i, err)
			}
		}
	})
	t.Run("IPv6", func(t *testing.T) {
		for i, tm := range []icmp.Message{
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
			{
				Type: ipv6.ICMPTypeExtendedEchoRequest, Code: 0,
				Body: &icmp.ExtendedEchoRequest{
					ID: 1, Seq: 2, Local: true,
					Extensions: []icmp.Extension{
						&icmp.InterfaceIdent{
							Class: 3,
							Type:  1,
							Name:  "en101",
						},
					},
				},
			},
			{
				Type: ipv6.ICMPTypeExtendedEchoRequest, Code: 0,
				Body: &icmp.ExtendedEchoRequest{
					ID: 1, Seq: 2, Local: true,
					Extensions: []icmp.Extension{
						&icmp.InterfaceIdent{
							Class: 3,
							Type:  2,
							Index: 911,
						},
					},
				},
			},
			{
				Type: ipv6.ICMPTypeExtendedEchoRequest, Code: 0,
				Body: &icmp.ExtendedEchoRequest{
					ID: 1, Seq: 2,
					Extensions: []icmp.Extension{
						&icmp.InterfaceIdent{
							Class: 3,
							Type:  3,
							AFI:   iana.AddrFamilyIPv4,
							Addr:  []byte{192, 0, 2, 1},
						},
					},
				},
			},
		} {
			if err := fn(t, iana.ProtocolIPv6ICMP, tm); err != nil {
				t.Errorf("#%d: %v", i, err)
			}
		}
	})
}

func dumpExtensions(gotExts, wantExts []icmp.Extension) string {
	var s string
	for i, got := range gotExts {
		switch got := got.(type) {
		case *icmp.MPLSLabelStack:
			want := wantExts[i].(*icmp.MPLSLabelStack)
			if !reflect.DeepEqual(got, want) {
				s += fmt.Sprintf("#%d: got %#v; want %#v\n", i, got, want)
			}
		case *icmp.InterfaceInfo:
			want := wantExts[i].(*icmp.InterfaceInfo)
			if !reflect.DeepEqual(got, want) {
				s += fmt.Sprintf("#%d: got %#v, %#v, %#v; want %#v, %#v, %#v\n", i, got, got.Interface, got.Addr, want, want.Interface, want.Addr)
			}
		case *icmp.InterfaceIdent:
			want := wantExts[i].(*icmp.InterfaceIdent)
			if !reflect.DeepEqual(got, want) {
				s += fmt.Sprintf("#%d: got %#v; want %#v\n", i, got, want)
			}
		case *icmp.RawExtension:
			s += fmt.Sprintf("#%d: raw extension\n", i)
		}
	}
	if len(s) == 0 {
		s += "empty extension"
	}
	return s[:len(s)-1]
}

func TestMultipartMessageBodyLen(t *testing.T) {
	for i, tt := range []struct {
		proto int
		in    icmp.MessageBody
		out   int
	}{
		{
			iana.ProtocolICMP,
			&icmp.DstUnreach{
				Data: make([]byte, ipv4.HeaderLen),
			},
			4 + ipv4.HeaderLen, // unused and original datagram
		},
		{
			iana.ProtocolICMP,
			&icmp.TimeExceeded{
				Data: make([]byte, ipv4.HeaderLen),
			},
			4 + ipv4.HeaderLen, // unused and original datagram
		},
		{
			iana.ProtocolICMP,
			&icmp.ParamProb{
				Data: make([]byte, ipv4.HeaderLen),
			},
			4 + ipv4.HeaderLen, // [pointer, unused] and original datagram
		},

		{
			iana.ProtocolICMP,
			&icmp.ParamProb{
				Data: make([]byte, ipv4.HeaderLen),
				Extensions: []icmp.Extension{
					&icmp.MPLSLabelStack{},
				},
			},
			4 + 4 + 4 + 0 + 128, // [pointer, length, unused], extension header, object header, object payload, original datagram
		},
		{
			iana.ProtocolICMP,
			&icmp.ParamProb{
				Data: make([]byte, 128),
				Extensions: []icmp.Extension{
					&icmp.MPLSLabelStack{},
				},
			},
			4 + 4 + 4 + 0 + 128, // [pointer, length, unused], extension header, object header, object payload and original datagram
		},
		{
			iana.ProtocolICMP,
			&icmp.ParamProb{
				Data: make([]byte, 129),
				Extensions: []icmp.Extension{
					&icmp.MPLSLabelStack{},
				},
			},
			4 + 4 + 4 + 0 + 132, // [pointer, length, unused], extension header, object header, object payload and original datagram
		},

		{
			iana.ProtocolIPv6ICMP,
			&icmp.DstUnreach{
				Data: make([]byte, ipv6.HeaderLen),
			},
			4 + ipv6.HeaderLen, // unused and original datagram
		},
		{
			iana.ProtocolIPv6ICMP,
			&icmp.PacketTooBig{
				Data: make([]byte, ipv6.HeaderLen),
			},
			4 + ipv6.HeaderLen, // mtu and original datagram
		},
		{
			iana.ProtocolIPv6ICMP,
			&icmp.TimeExceeded{
				Data: make([]byte, ipv6.HeaderLen),
			},
			4 + ipv6.HeaderLen, // unused and original datagram
		},
		{
			iana.ProtocolIPv6ICMP,
			&icmp.ParamProb{
				Data: make([]byte, ipv6.HeaderLen),
			},
			4 + ipv6.HeaderLen, // pointer and original datagram
		},

		{
			iana.ProtocolIPv6ICMP,
			&icmp.DstUnreach{
				Data: make([]byte, 127),
				Extensions: []icmp.Extension{
					&icmp.MPLSLabelStack{},
				},
			},
			4 + 4 + 4 + 0 + 128, // [length, unused], extension header, object header, object payload and original datagram
		},
		{
			iana.ProtocolIPv6ICMP,
			&icmp.DstUnreach{
				Data: make([]byte, 128),
				Extensions: []icmp.Extension{
					&icmp.MPLSLabelStack{},
				},
			},
			4 + 4 + 4 + 0 + 128, // [length, unused], extension header, object header, object payload and original datagram
		},
		{
			iana.ProtocolIPv6ICMP,
			&icmp.DstUnreach{
				Data: make([]byte, 129),
				Extensions: []icmp.Extension{
					&icmp.MPLSLabelStack{},
				},
			},
			4 + 4 + 4 + 0 + 136, // [length, unused], extension header, object header, object payload and original datagram
		},

		{
			iana.ProtocolICMP,
			&icmp.ExtendedEchoRequest{},
			4, // [id, seq, l-bit]
		},
		{
			iana.ProtocolICMP,
			&icmp.ExtendedEchoRequest{
				Extensions: []icmp.Extension{
					&icmp.InterfaceIdent{},
				},
			},
			4 + 4 + 4, // [id, seq, l-bit], extension header, object header
		},
		{
			iana.ProtocolIPv6ICMP,
			&icmp.ExtendedEchoRequest{
				Extensions: []icmp.Extension{
					&icmp.InterfaceIdent{
						Type: 3,
						AFI:  iana.AddrFamilyNSAP,
						Addr: []byte{0x49, 0x00, 0x01, 0xaa, 0xaa, 0xbb, 0xbb, 0xcc, 0xcc, 0x00},
					},
				},
			},
			4 + 4 + 4 + 16, // [id, seq, l-bit], extension header, object header, object payload
		},
	} {
		if out := tt.in.Len(tt.proto); out != tt.out {
			t.Errorf("#%d: got %d; want %d", i, out, tt.out)
		}
	}
}
