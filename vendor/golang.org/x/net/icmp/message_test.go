// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package icmp_test

import (
	"net"
	"reflect"
	"testing"

	"golang.org/x/net/icmp"
	"golang.org/x/net/internal/iana"
	"golang.org/x/net/ipv4"
	"golang.org/x/net/ipv6"
)

var marshalAndParseMessageForIPv4Tests = []icmp.Message{
	{
		Type: ipv4.ICMPTypeDestinationUnreachable, Code: 15,
		Body: &icmp.DstUnreach{
			Data: []byte("ERROR-INVOKING-PACKET"),
		},
	},
	{
		Type: ipv4.ICMPTypeTimeExceeded, Code: 1,
		Body: &icmp.TimeExceeded{
			Data: []byte("ERROR-INVOKING-PACKET"),
		},
	},
	{
		Type: ipv4.ICMPTypeParameterProblem, Code: 2,
		Body: &icmp.ParamProb{
			Pointer: 8,
			Data:    []byte("ERROR-INVOKING-PACKET"),
		},
	},
	{
		Type: ipv4.ICMPTypeEcho, Code: 0,
		Body: &icmp.Echo{
			ID: 1, Seq: 2,
			Data: []byte("HELLO-R-U-THERE"),
		},
	},
	{
		Type: ipv4.ICMPTypePhoturis,
		Body: &icmp.DefaultMessageBody{
			Data: []byte{0x80, 0x40, 0x20, 0x10},
		},
	},
}

func TestMarshalAndParseMessageForIPv4(t *testing.T) {
	for i, tt := range marshalAndParseMessageForIPv4Tests {
		b, err := tt.Marshal(nil)
		if err != nil {
			t.Fatal(err)
		}
		m, err := icmp.ParseMessage(iana.ProtocolICMP, b)
		if err != nil {
			t.Fatal(err)
		}
		if m.Type != tt.Type || m.Code != tt.Code {
			t.Errorf("#%v: got %v; want %v", i, m, &tt)
		}
		if !reflect.DeepEqual(m.Body, tt.Body) {
			t.Errorf("#%v: got %v; want %v", i, m.Body, tt.Body)
		}
	}
}

var marshalAndParseMessageForIPv6Tests = []icmp.Message{
	{
		Type: ipv6.ICMPTypeDestinationUnreachable, Code: 6,
		Body: &icmp.DstUnreach{
			Data: []byte("ERROR-INVOKING-PACKET"),
		},
	},
	{
		Type: ipv6.ICMPTypePacketTooBig, Code: 0,
		Body: &icmp.PacketTooBig{
			MTU:  1<<16 - 1,
			Data: []byte("ERROR-INVOKING-PACKET"),
		},
	},
	{
		Type: ipv6.ICMPTypeTimeExceeded, Code: 1,
		Body: &icmp.TimeExceeded{
			Data: []byte("ERROR-INVOKING-PACKET"),
		},
	},
	{
		Type: ipv6.ICMPTypeParameterProblem, Code: 2,
		Body: &icmp.ParamProb{
			Pointer: 8,
			Data:    []byte("ERROR-INVOKING-PACKET"),
		},
	},
	{
		Type: ipv6.ICMPTypeEchoRequest, Code: 0,
		Body: &icmp.Echo{
			ID: 1, Seq: 2,
			Data: []byte("HELLO-R-U-THERE"),
		},
	},
	{
		Type: ipv6.ICMPTypeDuplicateAddressConfirmation,
		Body: &icmp.DefaultMessageBody{
			Data: []byte{0x80, 0x40, 0x20, 0x10},
		},
	},
}

func TestMarshalAndParseMessageForIPv6(t *testing.T) {
	pshicmp := icmp.IPv6PseudoHeader(net.ParseIP("fe80::1"), net.ParseIP("ff02::1"))
	for i, tt := range marshalAndParseMessageForIPv6Tests {
		for _, psh := range [][]byte{pshicmp, nil} {
			b, err := tt.Marshal(psh)
			if err != nil {
				t.Fatal(err)
			}
			m, err := icmp.ParseMessage(iana.ProtocolIPv6ICMP, b)
			if err != nil {
				t.Fatal(err)
			}
			if m.Type != tt.Type || m.Code != tt.Code {
				t.Errorf("#%v: got %v; want %v", i, m, &tt)
			}
			if !reflect.DeepEqual(m.Body, tt.Body) {
				t.Errorf("#%v: got %v; want %v", i, m.Body, tt.Body)
			}
		}
	}
}
