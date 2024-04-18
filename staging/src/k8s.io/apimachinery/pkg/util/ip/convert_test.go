/*
Copyright 2024 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package ip

import (
	"net"
	"net/netip"
	"reflect"
	"testing"
)

func TestAddrFromIP_IPFromAddr(t *testing.T) {
	testCases := []struct {
		desc string
		ip   net.IP
		addr netip.Addr
	}{
		{
			desc: "IPv4 all-zeros",
			ip:   net.IPv4zero,
			addr: netip.IPv4Unspecified(),
		},
		{
			desc: "IPv6 all-zeros",
			ip:   net.IPv6zero,
			addr: netip.IPv6Unspecified(),
		},
		{
			desc: "IPv4 broadcast",
			ip:   net.IPv4bcast,
			addr: netip.AddrFrom4([4]byte{0xFF, 0xFF, 0xFF, 0xFF}),
		},
		{
			desc: "IPv4 loopback",
			ip:   net.IPv4(127, 0, 0, 1),
			addr: netip.AddrFrom4([4]byte{127, 0, 0, 1}),
		},
		{
			desc: "IPv6 loopback",
			ip:   net.IPv6loopback,
			addr: netip.IPv6Loopback(),
		},
		{
			desc: "IPv4 1",
			ip:   net.IPv4(10, 20, 40, 40),
			addr: netip.AddrFrom4([4]byte{10, 20, 40, 40}),
		},
		{
			desc: "IPv4 2",
			ip:   net.IPv4(172, 17, 3, 0),
			addr: netip.AddrFrom4([4]byte{172, 17, 3, 0}),
		},
		{
			desc: "IPv6 1",
			ip:   net.IP{0xFD, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x60, 0x0D, 0xF0, 0x0D},
			addr: netip.AddrFrom16([16]byte{0xFD, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x60, 0x0D, 0xF0, 0x0D}),
		},
		{
			desc: "IPv6 2",
			ip:   net.IP{0x20, 0x01, 0x0D, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x05},
			addr: netip.AddrFrom16([16]byte{0x20, 0x01, 0x0D, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x05}),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			addr := AddrFromIP(tc.ip)
			if addr != tc.addr {
				t.Errorf("AddrFromIP() expected %#v (%s) got %#v (%s)", tc.addr, tc.addr.String(), addr, addr.String())
			}

			ip := IPFromAddr(tc.addr)
			if !ip.Equal(tc.ip) {
				t.Errorf("IPFromAddr() expected %#v (%s) got %#v (%s)", tc.ip, tc.ip.String(), ip, ip.String())
			}
		})
	}

	// Special cases
	var ip, expectedIP net.IP
	var addr, expectedAddr netip.Addr

	// IPv4-mapped IPv6 gets converted to plain IPv4, in either direction
	ip = net.IP{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 1, 2, 3, 4}
	addr = AddrFromIP(ip)
	expectedAddr = netip.AddrFrom4([4]byte{1, 2, 3, 4})
	if addr != expectedAddr {
		t.Errorf("AddrFromIP(::ffff:1.2.3.4) expected %#v (%s) got %#v (%s)", expectedAddr, expectedAddr.String(), addr, addr.String())
	}

	addr = netip.AddrFrom16([16]byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 1, 2, 3, 4})
	ip = IPFromAddr(addr)
	expectedIP = net.IP{1, 2, 3, 4}
	if !ip.Equal(expectedIP) {
		t.Errorf("IPFromAddr(::ffff:1.2.3.4) expected %#v (%s) got %#v (%s)", expectedIP, expectedIP.String(), ip, ip.String())
	}

	// nil IP
	ip = nil
	addr = AddrFromIP(ip)
	expectedAddr = netip.Addr{}
	if addr != expectedAddr {
		t.Errorf("AddrFromIP(%s) expected %#v (%s) got %#v (%s)", ip.String(), expectedAddr, expectedAddr.String(), addr, addr.String())
	}
	ip = IPFromAddr(expectedAddr)
	if ip != nil {
		t.Errorf("IPFromAddr(%s) expected nil got %#v (%s)", expectedAddr.String(), ip, ip.String())
	}

	// invalid IP
	ip = net.IP{0x1}
	addr = AddrFromIP(ip)
	expectedAddr = netip.Addr{}
	if addr != expectedAddr {
		t.Errorf("AddrFromIP(%s) expected %#v (%s) got %#v (%s)", ip.String(), expectedAddr, expectedAddr.String(), addr, addr.String())
	}
	ip = IPFromAddr(expectedAddr)
	if ip != nil {
		t.Errorf("IPFromAddr(%s) expected nil got %#v (%s)", expectedAddr.String(), ip, ip.String())
	}
}

type dummyNetAddr string

func (d dummyNetAddr) Network() string {
	return "dummy"
}
func (d dummyNetAddr) String() string {
	return string(d)
}

func TestIPFromInterfaceAddr_AddrFromInterfaceAddr(t *testing.T) {
	testCases := []struct {
		desc   string
		ifaddr net.Addr
		out    string
	}{
		{
			desc:   "net.IPNet",
			ifaddr: &net.IPNet{IP: net.IP{192, 168, 1, 1}, Mask: net.CIDRMask(24, 32)},
			out:    "192.168.1.1",
		},
		{
			desc:   "net.IPAddr",
			ifaddr: &net.IPAddr{IP: net.IP{192, 168, 1, 2}},
			out:    "192.168.1.2",
		},
		{
			desc:   "net.IPAddr with zone",
			ifaddr: &net.IPAddr{IP: net.IP{192, 168, 1, 3}, Zone: "eth0"},
			out:    "192.168.1.3",
		},
		{
			desc:   "net.TCPAddr",
			ifaddr: &net.TCPAddr{IP: net.IP{192, 168, 1, 4}, Port: 80},
			out:    "",
		},
		{
			desc:   "unknown plain IP",
			ifaddr: dummyNetAddr("192.168.1.5"),
			out:    "192.168.1.5",
		},
		{
			desc:   "unknown CIDR",
			ifaddr: dummyNetAddr("192.168.1.6/24"),
			out:    "192.168.1.6",
		},
		{
			desc:   "unknown IP with zone",
			ifaddr: dummyNetAddr("192.168.1.7%eth0"),
			out:    "192.168.1.7",
		},
		{
			desc:   "unknown sockaddr",
			ifaddr: dummyNetAddr("192.168.1.8:80"),
			out:    "",
		},
		{
			desc:   "unknown junk",
			ifaddr: dummyNetAddr("junk"),
			out:    "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			ip := IPFromInterfaceAddr(tc.ifaddr)
			addr := AddrFromInterfaceAddr(tc.ifaddr)
			if tc.out == "" {
				if ip != nil {
					t.Errorf("expected IPFromInterfaceAddr to return nil but got %q", ip.String())
				}
				if addr.IsValid() {
					t.Errorf("expected AddrFromInterfaceAddr to return zero but got %q", addr.String())
				}
			} else {
				if ip.String() != tc.out {
					t.Errorf("expected IPFromInterfaceAddr to return %q but got %q", tc.out, ip.String())
				}
				if addr.String() != tc.out {
					t.Errorf("expected AddrFromInterfaceAddr to return %q but got %q", tc.out, addr.String())
				}
			}
		})
	}
}

func TestPrefixFromIPNet_IPNetFromPrefix(t *testing.T) {
	testCases := []struct {
		desc   string
		ipnet  *net.IPNet
		prefix netip.Prefix
	}{
		{
			desc: "IPv4 CIDR 1",
			ipnet: &net.IPNet{
				IP:   net.IPv4(10, 0, 0, 0),
				Mask: net.CIDRMask(8, 32),
			},
			prefix: netip.PrefixFrom(
				netip.AddrFrom4([4]byte{10, 0, 0, 0}),
				8,
			),
		},
		{
			desc: "IPv4 CIDR 2",
			ipnet: &net.IPNet{
				IP:   net.IPv4(192, 168, 0, 0),
				Mask: net.CIDRMask(16, 32),
			},
			prefix: netip.PrefixFrom(
				netip.AddrFrom4([4]byte{192, 168, 0, 0}),
				16,
			),
		},
		{
			desc: "IPv6 CIDR 1",
			ipnet: &net.IPNet{
				IP:   net.IPv6zero,
				Mask: net.CIDRMask(1, 128),
			},
			prefix: netip.PrefixFrom(
				netip.IPv6Unspecified(),
				1,
			),
		},
		{
			desc: "IPv6 CIDR 2",
			ipnet: &net.IPNet{
				IP:   net.IP{0x20, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				Mask: net.CIDRMask(10, 128),
			},
			prefix: netip.PrefixFrom(
				netip.AddrFrom16([16]byte{0x20, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
				10,
			),
		},
		{
			desc: "IPv6 CIDR 3",
			ipnet: &net.IPNet{
				IP:   net.IP{0x20, 0x01, 0x0D, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				Mask: net.CIDRMask(32, 128),
			},
			prefix: netip.PrefixFrom(
				netip.AddrFrom16([16]byte{0x20, 0x01, 0x0D, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
				32,
			),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			prefix := PrefixFromIPNet(tc.ipnet)
			if prefix != tc.prefix {
				t.Errorf("PrefixFromIPNet() expected %#v (%s) got %#v (%s)", tc.prefix, tc.prefix.String(), prefix, prefix.String())
			}

			ipnet := IPNetFromPrefix(tc.prefix)
			if !ipnet.IP.Equal(tc.ipnet.IP) || !reflect.DeepEqual(ipnet.Mask, tc.ipnet.Mask) {
				t.Errorf("IPNetFromPrefix() expected %#v (%s) got %#v (%s)", tc.ipnet, tc.ipnet.String(), ipnet, ipnet.String())
			}
		})
	}

	// Special cases
	var ipnet *net.IPNet
	var prefix, expectedPrefix netip.Prefix

	// If you call net.ParseCIDR("1.2.3.0/24") you'd get a 4-byte IP and a 4-byte Mask
	// like in "IPv4 CIDR 1" above. But if you construct an IPNet by hand with a
	// parsed IPv4 IP:
	//
	//    &net.IPNet{IP: net.ParseIP("1.2.3.0"), Mask: net.CIDRMask(24, 32)}
	//
	// you'd get a 16-byte IP and a 4-byte Mask). However, IPNet.String() and
	// IP.Mask() treat this the same as the 4-byte case, so we do too.
	ipnet = &net.IPNet{
		IP:   net.IP{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 1, 2, 3, 0},
		Mask: net.CIDRMask(24, 32),
	}
	prefix = PrefixFromIPNet(ipnet)
	expectedPrefix = netip.PrefixFrom(netip.AddrFrom4([4]byte{1, 2, 3, 0}), 24)
	if prefix != expectedPrefix {
		t.Errorf("PrefixFromIPNet(IPv4-mapped IPv6) expected %#v (%s) got %#v (%s)", expectedPrefix, expectedPrefix.String(), prefix, prefix.String())
	}

	// Similarly, if you call net.ParseCIDR("::ffff:1.2.3.0/120"), you get a 16-byte
	// IP and a 16-byte mask, but again it still behaves the same as in the 4-byte
	// case. (There is no reason anyone *should* do this, but we support the
	// conversion for backward-compatibility.)
	ipnet = &net.IPNet{
		IP:   net.IP{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 1, 2, 3, 0},
		Mask: net.CIDRMask(120, 128),
	}
	prefix = PrefixFromIPNet(ipnet)
	if prefix != expectedPrefix {
		t.Errorf("PrefixFromIPNet(IPv4-mapped IPv6) expected %#v (%s) got %#v (%s)", expectedPrefix, expectedPrefix.String(), prefix, prefix.String())
	}

	// OTOH, there's no good reason to try to support the conversion in the other
	// direction, since, as noted in convert.go, netip.Prefix is basically broken in
	// this case.
	prefix = netip.PrefixFrom(
		netip.AddrFrom16([16]byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 1, 2, 3, 0}),
		120,
	)
	ipnet = IPNetFromPrefix(prefix)
	if ipnet != nil {
		t.Errorf("IPNetFromPrefix(IPv4-mapped IPv6) expected nil got %#v (%s)", ipnet, ipnet.String())
	}

	// nil IPNet
	ipnet = nil
	prefix = PrefixFromIPNet(ipnet)
	expectedPrefix = netip.Prefix{}
	if prefix != expectedPrefix {
		t.Errorf("PrefixFromIPNet(nil) expected %#v (%s) got %#v (%s)", expectedPrefix, expectedPrefix.String(), prefix, prefix.String())
	}
	ipnet = IPNetFromPrefix(expectedPrefix)
	if ipnet != nil {
		t.Errorf("IPNetFromPrefix(zero) expected nil got %#v (%s)", ipnet, ipnet.String())
	}

	// invalid IPNet
	ipnet = &net.IPNet{IP: net.IP{0x1}}
	prefix = PrefixFromIPNet(ipnet)
	expectedPrefix = netip.Prefix{}
	if prefix != expectedPrefix {
		t.Errorf("PrefixFromIPNet(invalid) expected %#v (%s) got %#v (%s)", expectedPrefix, expectedPrefix.String(), prefix, prefix.String())
	}
}
