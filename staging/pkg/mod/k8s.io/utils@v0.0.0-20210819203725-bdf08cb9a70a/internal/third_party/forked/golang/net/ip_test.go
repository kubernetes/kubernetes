// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

///////////////////////////////////////////////////////////////////////////////
// NOTE: This file was forked because we need to maintain backwards-compatible
// IP parsing logic, which was changed in a correct but incompatible way in
// go-1.17.
//
// See https://issue.k8s.io/100895
///////////////////////////////////////////////////////////////////////////////

import (
	stdnet "net"
	"reflect"
	"testing"
)

//
// Lean on the standard net lib as much as possible.
//
type IPMask = stdnet.IPMask

var IPv4Mask = stdnet.IPv4Mask

var parseIPTests = []struct {
	in  string
	out IP
}{
	{"127.0.1.2", IPv4(127, 0, 1, 2)},
	{"127.0.0.1", IPv4(127, 0, 0, 1)},
	{"127.001.002.003", IPv4(127, 1, 2, 3)},    // see https://issue.k8s.io/100895
	{"127.007.008.009", IPv4(127, 7, 8, 9)},    // see https://issue.k8s.io/100895
	{"127.010.020.030", IPv4(127, 10, 20, 30)}, // see https://issue.k8s.io/100895
	{"::ffff:127.1.2.3", IPv4(127, 1, 2, 3)},
	{"::ffff:127.001.002.003", IPv4(127, 1, 2, 3)},    // see https://issue.k8s.io/100895
	{"::ffff:127.007.008.009", IPv4(127, 7, 8, 9)},    // see https://issue.k8s.io/100895
	{"::ffff:127.010.020.030", IPv4(127, 10, 20, 30)}, // see https://issue.k8s.io/100895
	{"::ffff:7f01:0203", IPv4(127, 1, 2, 3)},
	{"0:0:0:0:0000:ffff:127.1.2.3", IPv4(127, 1, 2, 3)},
	{"0:0:0:0:000000:ffff:127.1.2.3", IPv4(127, 1, 2, 3)},
	{"0:0:0:0::ffff:127.1.2.3", IPv4(127, 1, 2, 3)},

	{"2001:4860:0:2001::68", IP{0x20, 0x01, 0x48, 0x60, 0, 0, 0x20, 0x01, 0, 0, 0, 0, 0, 0, 0x00, 0x68}},
	{"2001:4860:0000:2001:0000:0000:0000:0068", IP{0x20, 0x01, 0x48, 0x60, 0, 0, 0x20, 0x01, 0, 0, 0, 0, 0, 0, 0x00, 0x68}},

	{"-0.0.0.0", nil},
	{"0.-1.0.0", nil},
	{"0.0.-2.0", nil},
	{"0.0.0.-3", nil},
	{"127.0.0.256", nil},
	{"abc", nil},
	{"123:", nil},
	{"fe80::1%lo0", nil},
	{"fe80::1%911", nil},
	{"", nil},
	{"a1:a2:a3:a4::b1:b2:b3:b4", nil}, // Issue 6628
	//
	// NOTE: These correct failures were added for go-1.17, but are a
	// backwards-incompatible change for kubernetes users, who might have
	// stored data which uses these leading zeroes already.
	//
	// See https://github.com/kubernetes/kubernetes/issues/100895
	//
	//{"127.001.002.003", nil},
	//{"::ffff:127.001.002.003", nil},
	//{"123.000.000.000", nil},
	//{"1.2..4", nil},
	//{"0123.0.0.1", nil},
}

func TestParseIP(t *testing.T) {
	for _, tt := range parseIPTests {
		if out := ParseIP(tt.in); !reflect.DeepEqual(out, tt.out) {
			t.Errorf("ParseIP(%q) = %v, want %v", tt.in, out, tt.out)
		}
		if tt.in == "" {
			// Tested in TestMarshalEmptyIP below.
			continue
		}
		var out IP
		if err := out.UnmarshalText([]byte(tt.in)); !reflect.DeepEqual(out, tt.out) || (tt.out == nil) != (err != nil) {
			t.Errorf("IP.UnmarshalText(%q) = %v, %v, want %v", tt.in, out, err, tt.out)
		}
	}
}

var parseCIDRTests = []struct {
	in  string
	ip  IP
	net *IPNet
	err error
}{
	{"135.104.0.0/32", IPv4(135, 104, 0, 0), &IPNet{IP: IPv4(135, 104, 0, 0), Mask: IPv4Mask(255, 255, 255, 255)}, nil},
	{"0.0.0.0/24", IPv4(0, 0, 0, 0), &IPNet{IP: IPv4(0, 0, 0, 0), Mask: IPv4Mask(255, 255, 255, 0)}, nil},
	{"135.104.0.0/24", IPv4(135, 104, 0, 0), &IPNet{IP: IPv4(135, 104, 0, 0), Mask: IPv4Mask(255, 255, 255, 0)}, nil},
	{"135.104.0.1/32", IPv4(135, 104, 0, 1), &IPNet{IP: IPv4(135, 104, 0, 1), Mask: IPv4Mask(255, 255, 255, 255)}, nil},
	{"135.104.0.1/24", IPv4(135, 104, 0, 1), &IPNet{IP: IPv4(135, 104, 0, 0), Mask: IPv4Mask(255, 255, 255, 0)}, nil},
	// see https://issue.k8s.io/100895
	{"127.000.000.001/32", IPv4(127, 0, 0, 1), &IPNet{IP: IPv4(127, 0, 0, 1), Mask: IPv4Mask(255, 255, 255, 255)}, nil},
	{"127.007.008.009/32", IPv4(127, 7, 8, 9), &IPNet{IP: IPv4(127, 7, 8, 9), Mask: IPv4Mask(255, 255, 255, 255)}, nil},
	{"127.010.020.030/32", IPv4(127, 10, 20, 30), &IPNet{IP: IPv4(127, 10, 20, 30), Mask: IPv4Mask(255, 255, 255, 255)}, nil},
	{"::1/128", ParseIP("::1"), &IPNet{IP: ParseIP("::1"), Mask: IPMask(ParseIP("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"))}, nil},
	{"abcd:2345::/127", ParseIP("abcd:2345::"), &IPNet{IP: ParseIP("abcd:2345::"), Mask: IPMask(ParseIP("ffff:ffff:ffff:ffff:ffff:ffff:ffff:fffe"))}, nil},
	{"abcd:2345::/65", ParseIP("abcd:2345::"), &IPNet{IP: ParseIP("abcd:2345::"), Mask: IPMask(ParseIP("ffff:ffff:ffff:ffff:8000::"))}, nil},
	{"abcd:2345::/64", ParseIP("abcd:2345::"), &IPNet{IP: ParseIP("abcd:2345::"), Mask: IPMask(ParseIP("ffff:ffff:ffff:ffff::"))}, nil},
	{"abcd:2345::/63", ParseIP("abcd:2345::"), &IPNet{IP: ParseIP("abcd:2345::"), Mask: IPMask(ParseIP("ffff:ffff:ffff:fffe::"))}, nil},
	{"abcd:2345::/33", ParseIP("abcd:2345::"), &IPNet{IP: ParseIP("abcd:2345::"), Mask: IPMask(ParseIP("ffff:ffff:8000::"))}, nil},
	{"abcd:2345::/32", ParseIP("abcd:2345::"), &IPNet{IP: ParseIP("abcd:2345::"), Mask: IPMask(ParseIP("ffff:ffff::"))}, nil},
	{"abcd:2344::/31", ParseIP("abcd:2344::"), &IPNet{IP: ParseIP("abcd:2344::"), Mask: IPMask(ParseIP("ffff:fffe::"))}, nil},
	{"abcd:2300::/24", ParseIP("abcd:2300::"), &IPNet{IP: ParseIP("abcd:2300::"), Mask: IPMask(ParseIP("ffff:ff00::"))}, nil},
	{"abcd:2345::/24", ParseIP("abcd:2345::"), &IPNet{IP: ParseIP("abcd:2300::"), Mask: IPMask(ParseIP("ffff:ff00::"))}, nil},
	{"2001:DB8::/48", ParseIP("2001:DB8::"), &IPNet{IP: ParseIP("2001:DB8::"), Mask: IPMask(ParseIP("ffff:ffff:ffff::"))}, nil},
	{"2001:DB8::1/48", ParseIP("2001:DB8::1"), &IPNet{IP: ParseIP("2001:DB8::"), Mask: IPMask(ParseIP("ffff:ffff:ffff::"))}, nil},
	{"192.168.1.1/255.255.255.0", nil, nil, &ParseError{Type: "CIDR address", Text: "192.168.1.1/255.255.255.0"}},
	{"192.168.1.1/35", nil, nil, &ParseError{Type: "CIDR address", Text: "192.168.1.1/35"}},
	{"2001:db8::1/-1", nil, nil, &ParseError{Type: "CIDR address", Text: "2001:db8::1/-1"}},
	{"2001:db8::1/-0", nil, nil, &ParseError{Type: "CIDR address", Text: "2001:db8::1/-0"}},
	{"-0.0.0.0/32", nil, nil, &ParseError{Type: "CIDR address", Text: "-0.0.0.0/32"}},
	{"0.-1.0.0/32", nil, nil, &ParseError{Type: "CIDR address", Text: "0.-1.0.0/32"}},
	{"0.0.-2.0/32", nil, nil, &ParseError{Type: "CIDR address", Text: "0.0.-2.0/32"}},
	{"0.0.0.-3/32", nil, nil, &ParseError{Type: "CIDR address", Text: "0.0.0.-3/32"}},
	{"0.0.0.0/-0", nil, nil, &ParseError{Type: "CIDR address", Text: "0.0.0.0/-0"}},
	// see https://issue.k8s.io/100895
	//{"127.000.000.001/32", nil, nil, &ParseError{Type: "CIDR address", Text: "127.000.000.001/32"}},
	{"", nil, nil, &ParseError{Type: "CIDR address", Text: ""}},
}

func TestParseCIDR(t *testing.T) {
	for _, tt := range parseCIDRTests {
		ip, net, err := ParseCIDR(tt.in)
		if !reflect.DeepEqual(err, tt.err) {
			t.Errorf("ParseCIDR(%q) = %v, %v; want %v, %v", tt.in, ip, net, tt.ip, tt.net)
		}
		if err == nil && (!tt.ip.Equal(ip) || !tt.net.IP.Equal(net.IP) || !reflect.DeepEqual(net.Mask, tt.net.Mask)) {
			t.Errorf("ParseCIDR(%q) = %v, {%v, %v}; want %v, {%v, %v}", tt.in, ip, net.IP, net.Mask, tt.ip, tt.net.IP, tt.net.Mask)
		}
	}
}
