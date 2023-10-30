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
	"testing"

	netutils "k8s.io/utils/net"
)

// This tests IPFamilyOf() against various net.IP and netip.Addr values, along with the
// corresponding canonical string values. Tests against non-canonical and invalid string
// values are below in TestIPFamilyOfString()
func TestIPFamilyOf(t *testing.T) {
	testCases := []struct {
		desc   string
		ip     net.IP
		addr   netip.Addr
		str    string
		family IPFamily
	}{
		{
			desc:   "IPv4 all-zeros",
			ip:     net.IPv4zero,
			addr:   netip.IPv4Unspecified(),
			str:    "0.0.0.0",
			family: IPv4Protocol,
		},
		{
			desc:   "IPv6 all-zeros",
			ip:     net.IPv6zero,
			addr:   netip.IPv6Unspecified(),
			str:    "::",
			family: IPv6Protocol,
		},
		{
			desc:   "IPv4 broadcast",
			ip:     net.IPv4bcast,
			addr:   netip.AddrFrom4([4]byte{0xFF, 0xFF, 0xFF, 0xFF}),
			str:    "255.255.255.255",
			family: IPv4Protocol,
		},
		{
			desc:   "IPv4 loopback",
			ip:     net.IPv4(127, 0, 0, 1),
			addr:   netip.AddrFrom4([4]byte{127, 0, 0, 1}),
			str:    "127.0.0.1",
			family: IPv4Protocol,
		},
		{
			desc:   "IPv6 loopback",
			ip:     net.IPv6loopback,
			addr:   netip.IPv6Loopback(),
			str:    "::1",
			family: IPv6Protocol,
		},
		{
			desc:   "IPv4 1",
			ip:     net.IPv4(10, 20, 40, 40),
			addr:   netip.AddrFrom4([4]byte{10, 20, 40, 40}),
			str:    "10.20.40.40",
			family: IPv4Protocol,
		},
		{
			desc:   "IPv4 2",
			ip:     net.IPv4(172, 17, 3, 0),
			addr:   netip.AddrFrom4([4]byte{172, 17, 3, 0}),
			str:    "172.17.3.0",
			family: IPv4Protocol,
		},
		{
			desc: "IPv4 encoded as IPv6 is IPv4",
			ip:   net.IP{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 1, 2, 3, 4},
			addr: netip.AddrFrom16([16]byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 1, 2, 3, 4}),
			// The netip.Addr version stringifies to "::ffff:1.2.3.4" but the
			// net.IP version stringifies to just "1.2.3.4".
			str:    "",
			family: IPv4Protocol,
		},
		{
			desc:   "IPv6 1",
			ip:     net.IP{0xFD, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x60, 0x0D, 0xF0, 0x0D},
			addr:   netip.AddrFrom16([16]byte{0xFD, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x60, 0x0D, 0xF0, 0x0D}),
			str:    "fd00::600d:f00d",
			family: IPv6Protocol,
		},
		{
			desc:   "IPv6 2",
			ip:     net.IP{0x20, 0x01, 0x0D, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x05},
			addr:   netip.AddrFrom16([16]byte{0x20, 0x01, 0x0D, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x05}),
			str:    "2001:db8::5",
			family: IPv6Protocol,
		},
		{
			desc:   "zero IP",
			ip:     nil,
			addr:   netip.Addr{},
			str:    "",
			family: IPFamilyUnknown,
		},
		{
			desc: "invalid",
			ip:   net.IP{0x1},
			// There is no way to generate an invalid netip.Addr other than
			// the zero Addr, so we just redundantly test that again...
			addr:   netip.Addr{},
			str:    "",
			family: IPFamilyUnknown,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			family := IPFamilyOf(tc.ip)
			isIPv4 := IsIPv4(tc.ip)
			isIPv6 := IsIPv6(tc.ip)
			if family != tc.family {
				t.Errorf("Expect family %q, net.IP got %q", tc.family, family)
			}
			if isIPv4 != (tc.family == IPv4Protocol) {
				t.Errorf("Expect ipv4 %v, net.IP got %v", tc.family == IPv4Protocol, isIPv6)
			}
			if isIPv6 != (tc.family == IPv6Protocol) {
				t.Errorf("Expect ipv6 %v, net.IP got %v", tc.family == IPv6Protocol, isIPv6)
			}

			family = IPFamilyOf(tc.addr)
			if family != tc.family {
				t.Errorf("Expect family %q, netip.Addr got %q", tc.family, family)
			}
			isIPv4 = IsIPv4(tc.addr)
			isIPv6 = IsIPv6(tc.addr)
			if isIPv4 != (tc.family == IPv4Protocol) {
				t.Errorf("Expect ipv4 %v, netip.Addr got %v", tc.family == IPv4Protocol, isIPv6)
			}
			if isIPv6 != (tc.family == IPv6Protocol) {
				t.Errorf("Expect ipv6 %v, netip.Addr got %v", tc.family == IPv6Protocol, isIPv6)
			}

			if tc.str != "" {
				family = IPFamilyOf(tc.str)
				if family != tc.family {
					t.Errorf("Expect family %q, str got %q", tc.family, family)
				}
				isIPv4 = IsIPv4(tc.str)
				isIPv6 = IsIPv6(tc.str)
				if isIPv4 != (tc.family == IPv4Protocol) {
					t.Errorf("Expect ipv4 %v, str got %v", tc.family == IPv4Protocol, isIPv6)
				}
				if isIPv6 != (tc.family == IPv6Protocol) {
					t.Errorf("Expect ipv6 %v, str got %v", tc.family == IPv6Protocol, isIPv6)
				}

				str := tc.ip.String()
				if str != tc.str {
					t.Errorf("Expect net.IP.String() %q, got %q", tc.str, str)
				}
				str = tc.addr.String()
				if str != tc.str {
					t.Errorf("Expect netip.Addr.String() %q, got %q", tc.str, str)
				}
			}
		})
	}
}

// Further IPFamilyOf tests that don't correspond 1-1 to net.IP/netip.Addr tests
func TestIPFamilyOfString(t *testing.T) {
	testCases := []struct {
		desc   string
		str    string
		family IPFamily
	}{
		// Good but weird
		{
			desc:   "IPv4 with leading 0s is accepted",
			str:    "001.002.003.004",
			family: IPv4Protocol,
		},
		{
			desc:   "IPv6 with extra 0s is accepted",
			str:    "2001:0db8:0000:0000:0000:0000:0000:0005",
			family: IPv6Protocol,
		},
		{
			desc:   "IPv6 with capital letters is accepted",
			str:    "2001:DB8::5",
			family: IPv6Protocol,
		},
		{
			desc:   "IPv4 encoded as IPv6 is IPv4",
			str:    "::ffff:1.2.3.4",
			family: IPv4Protocol,
		},

		// Bad
		{
			desc:   "empty string is invalid",
			str:    "",
			family: IPFamilyUnknown,
		},
		{
			desc:   "random unparseable string is invalid",
			str:    "bad ip",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv4 with out-of-range octets is invalid",
			str:    "1.2.3.400",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv6 with out-of-range segment is invalid",
			str:    "2001:db8::10005",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv4 with empty octet is invalid",
			str:    "1.2..4",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv6 with multiple empty segments is invalid",
			str:    "2001::db8::5",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv4 CIDR is invalid",
			str:    "1.2.3.4/32",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv6 CIDR is invalid",
			str:    "2001:db8::/64",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv4:port is invalid",
			str:    "1.2.3.4:80",
			family: IPFamilyUnknown,
		},
		{
			desc:   "[IPv6] with brackets is invalid",
			str:    "[2001:db8::5]",
			family: IPFamilyUnknown,
		},
		{
			desc:   "[IPv6]:port is invalid",
			str:    "[2001:db8::5]:80",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv6%zone is invalid",
			str:    "fe80::1234%eth0",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv4 with leading whitespace is invalid",
			str:    " 1.2.3.4",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv4 with trailing whitespace is invalid",
			str:    "1.2.3.4 ",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv6 with leading whitespace is invalid",
			str:    " 2001:db8::5",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv6 with trailing whitespace is invalid",
			str:    "2001:db8::5 ",
			family: IPFamilyUnknown,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			family := IPFamilyOf(tc.str)
			isIPv4 := IsIPv4(tc.str)
			isIPv6 := IsIPv6(tc.str)

			if family != tc.family {
				t.Errorf("Expect %q family %q, got %q", tc.str, tc.family, family)
			}
			if isIPv4 != (tc.family == IPv4Protocol) {
				t.Errorf("Expect %q ipv4 %v, got %v", tc.str, tc.family == IPv4Protocol, isIPv6)
			}
			if isIPv6 != (tc.family == IPv6Protocol) {
				t.Errorf("Expect %q ipv6 %v, got %v", tc.str, tc.family == IPv6Protocol, isIPv6)
			}
		})
	}
}

// This tests IPFamilyOfCIDR() against various *net.IPNet and netip.Prefix cases, along
// with the correspondings canonical string values. Tests against non-canonical and
// invalid string values are below in TestIPFamilyOfCIDRString().
func TestIPFamilyOfCIDR(t *testing.T) {
	testCases := []struct {
		desc   string
		ipnet  *net.IPNet
		prefix netip.Prefix
		str    string
		family IPFamily
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
			str:    "10.0.0.0/8",
			family: IPv4Protocol,
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
			str:    "192.168.0.0/16",
			family: IPv4Protocol,
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
			str:    "::/1",
			family: IPv6Protocol,
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
			str:    "2000::/10",
			family: IPv6Protocol,
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
			str:    "2001:db8::/32",
			family: IPv6Protocol,
		},
		{
			desc:   "zero CIDR",
			ipnet:  nil,
			prefix: netip.Prefix{},
			str:    "",
			family: IPFamilyUnknown,
		},
		{
			desc: "invalid",
			ipnet: &net.IPNet{
				IP: net.IP{0x1},
			},
			// There is no way to generate an invalid netip.Prefix other than
			// the zero Prefix, so we just redundantly test that again...
			prefix: netip.Prefix{},
			str:    "",
			family: IPFamilyUnknown,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			family := IPFamilyOfCIDR(tc.ipnet)
			if family != tc.family {
				t.Errorf("Expect family %v, *net.IPNet got %v", tc.family, family)
			}
			isIPv4 := IsIPv4CIDR(tc.ipnet)
			isIPv6 := IsIPv6CIDR(tc.ipnet)
			if isIPv4 != (tc.family == IPv4Protocol) {
				t.Errorf("Expect ipv4 %v, *net.IPNet got %v", tc.family == IPv4Protocol, isIPv6)
			}
			if isIPv6 != (tc.family == IPv6Protocol) {
				t.Errorf("Expect ipv6 %v, *net.IPNet got %v", tc.family == IPv6Protocol, isIPv6)
			}

			family = IPFamilyOfCIDR(tc.prefix)
			if family != tc.family {
				t.Errorf("Expect family %v, netip.Prefix got %v", tc.family, family)
			}
			isIPv4 = IsIPv4CIDR(tc.prefix)
			isIPv6 = IsIPv6CIDR(tc.prefix)
			if isIPv4 != (tc.family == IPv4Protocol) {
				t.Errorf("Expect ipv4 %v, netip.Prefix got %v", tc.family == IPv4Protocol, isIPv6)
			}
			if isIPv6 != (tc.family == IPv6Protocol) {
				t.Errorf("Expect ipv6 %v, netip.Prefix got %v", tc.family == IPv6Protocol, isIPv6)
			}

			if tc.str != "" {
				family = IPFamilyOfCIDR(tc.str)
				if family != tc.family {
					t.Errorf("Expect family %v, str got %v", tc.family, family)
				}
				isIPv4 = IsIPv4CIDR(tc.str)
				isIPv6 = IsIPv6CIDR(tc.str)
				if isIPv4 != (tc.family == IPv4Protocol) {
					t.Errorf("Expect ipv4 %v, str got %v", tc.family == IPv4Protocol, isIPv6)
				}
				if isIPv6 != (tc.family == IPv6Protocol) {
					t.Errorf("Expect ipv6 %v, str got %v", tc.family == IPv6Protocol, isIPv6)
				}

				str := tc.ipnet.String()
				if str != tc.str {
					t.Errorf("Expect *net.IPNet.String() %q, got %q", tc.str, str)
				}
				str = tc.prefix.String()
				if str != tc.str {
					t.Errorf("Expect netip.Prefix.String() %q, got %q", tc.str, str)
				}
			}
		})
	}
}

// Further IPFamilyOfCIDR tests that don't correspond 1-1 to *net.IPNet/netip.Prefix tests
func TestIPFamilyOfCIDRString(t *testing.T) {
	testCases := []struct {
		desc   string
		str    string
		family IPFamily
	}{
		// Good but weird
		{
			desc:   "IPv4 with leading 0s is accepted",
			str:    "001.002.003.000/24",
			family: IPv4Protocol,
		},
		{
			desc:   "IPv4 with address bits beyond mask is accepted",
			str:    "1.2.3.4/24",
			family: IPv4Protocol,
		},
		{
			desc:   "IPv6 with extra 0s is accepted",
			str:    "2001:0db8:0000:0000:0000:0000:0000:0000/64",
			family: IPv6Protocol,
		},
		{
			desc:   "IPv6 with capital letters is accepted",
			str:    "2001:DB8::/64",
			family: IPv6Protocol,
		},
		{
			desc:   "IPv6 with address bits beyond mask is accepted",
			str:    "2001:db8::5/64",
			family: IPv6Protocol,
		},

		// Bad
		{
			desc:   "empty string is invalid",
			str:    "",
			family: IPFamilyUnknown,
		},
		{
			desc:   "random unparseable string is invalid",
			str:    "bad/cidr",
			family: IPFamilyUnknown,
		},
		{
			desc:   "bad CIDR",
			str:    "foo",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv4 with out-of-range octets is invalid",
			str:    "1.2.3.400/32",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv6 with out-of-range segment is invalid",
			str:    "2001:db8::10005/64",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv4 with out-of-range mask length is invalid",
			str:    "1.2.3.4/64",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv6 with out-of-range mask length is invalid",
			str:    "2001:db8::5/192",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv4 with empty octet is invalid",
			str:    "1.2..4/32",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv6 with multiple empty segments is invalid",
			str:    "2001::db8::5/64",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv4 IP is not CIDR",
			str:    "192.168.0.0",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv6 IP is not CIDR",
			str:    "2001:db8::",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv4 CIDR with leading whitespace is invalid",
			str:    " 1.2.3.4/32",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv4 CIDR with trailing whitespace is invalid",
			str:    "1.2.3.4/32 ",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv6 CIDR with leading whitespace is invalid",
			str:    " 2001:db8::5/64",
			family: IPFamilyUnknown,
		},
		{
			desc:   "IPv6 CIDR with trailing whitespace is invalid",
			str:    "2001:db8::5/64 ",
			family: IPFamilyUnknown,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			family := IPFamilyOfCIDR(tc.str)
			isIPv4 := IsIPv4CIDR(tc.str)
			isIPv6 := IsIPv6CIDR(tc.str)

			if family != tc.family {
				t.Errorf("Expect family %v, got %v", tc.family, family)
			}
			if isIPv4 != (tc.family == IPv4Protocol) {
				t.Errorf("Expect %q ipv4 %v, got %v", tc.str, tc.family == IPv4Protocol, isIPv6)
			}
			if isIPv6 != (tc.family == IPv6Protocol) {
				t.Errorf("Expect %q ipv6 %v, got %v", tc.str, tc.family == IPv6Protocol, isIPv6)
			}
		})
	}
}

func TestDualStackIPs(t *testing.T) {
	testCases := []struct {
		desc      string
		ips       []string
		dualStack bool
	}{
		{
			desc:      "false because length is not at least 2",
			ips:       []string{"1.1.1.1"},
			dualStack: false,
		},
		{
			desc:      "false because length is not at least 2",
			ips:       []string{},
			dualStack: false,
		},
		{
			desc:      "false because all are v4",
			ips:       []string{"1.1.1.1", "2.2.2.2", "3.3.3.3"},
			dualStack: false,
		},
		{
			desc:      "false because all are v6",
			ips:       []string{"fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff", "fd92:20ba:ca:34f7:ffff:ffff:ffff:fff0", "fd92:20ba:ca:34f7:ffff:ffff:ffff:fff1"},
			dualStack: false,
		},
		{
			desc:      "false because 2nd ip is invalid",
			ips:       []string{"1.1.1.1", "not-a-valid-ip"},
			dualStack: false,
		},
		{
			desc:      "false because 1st ip is invalid",
			ips:       []string{"not-a-valid-ip", "fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff"},
			dualStack: false,
		},
		{
			desc:      "false despite dual-stack IPs because 3rd IP is invalid",
			ips:       []string{"1.1.1.1", "fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff", "not-a-valid-ip"},
			dualStack: false,
		},
		{
			desc:      "valid dual-stack",
			ips:       []string{"1.1.1.1", "fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff"},
			dualStack: true,
		},
		{
			desc:      "valid dual stack with multiple IPv6",
			ips:       []string{"fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff", "1.1.1.1", "fd92:20ba:ca:34f7:ffff:ffff:ffff:fff0"},
			dualStack: true,
		},
		{
			desc:      "valid dual stack with multiple IPv4",
			ips:       []string{"1.1.1.1", "fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff", "10.0.0.0"},
			dualStack: true,
		},
		{
			desc:      "valid dual-stack, IPv6-primary",
			ips:       []string{"fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff", "1.1.1.1"},
			dualStack: true,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			netips := make([]net.IP, len(tc.ips))
			addrs := make([]netip.Addr, len(tc.ips))
			for i := range tc.ips {
				netips[i] = netutils.ParseIPSloppy(tc.ips[i])
				addrs[i], _ = netip.ParseAddr(tc.ips[i])
			}

			dualStack := IsDualStackIPs(tc.ips)
			if dualStack != tc.dualStack {
				t.Errorf("expected %v, []string got %v", tc.dualStack, dualStack)
			}
			if IsDualStackIPPair(tc.ips) != (dualStack && len(tc.ips) == 2) {
				t.Errorf("IsDualStackIPPair gave wrong result for []string")
			}

			dualStack = IsDualStackIPs(netips)
			if dualStack != tc.dualStack {
				t.Errorf("expected %v []net.IP got %v", tc.dualStack, dualStack)
			}
			if IsDualStackIPPair(netips) != (dualStack && len(tc.ips) == 2) {
				t.Errorf("IsDualStackIPPair gave wrong result for []net.IP")
			}

			dualStack = IsDualStackIPs(addrs)
			if dualStack != tc.dualStack {
				t.Errorf("expected %v []netip.Addr got %v", tc.dualStack, dualStack)
			}
			if IsDualStackIPPair(addrs) != (dualStack && len(tc.ips) == 2) {
				t.Errorf("IsDualStackIPPair gave wrong result for []netip.Addr")
			}
		})
	}
}

func TestDualStackCIDRs(t *testing.T) {
	testCases := []struct {
		desc      string
		cidrs     []string
		dualStack bool
	}{
		{
			desc:      "false because length is not at least 2",
			cidrs:     []string{"10.0.0.0/8"},
			dualStack: false,
		},
		{
			desc:      "false because length is not at least 2",
			cidrs:     []string{},
			dualStack: false,
		},
		{
			desc:      "false because all cidrs are v4",
			cidrs:     []string{"10.0.0.0/8", "20.0.0.0/8", "30.0.0.0/8"},
			dualStack: false,
		},
		{
			desc:      "false because all cidrs are v6",
			cidrs:     []string{"2000::/10", "3000::/10"},
			dualStack: false,
		},
		{
			desc:      "false because 2nd cidr is invalid",
			cidrs:     []string{"10.0.0.0/8", "not-a-valid-cidr"},
			dualStack: false,
		},
		{
			desc:      "false because 1st cidr is invalid",
			cidrs:     []string{"not-a-valid-ip", "2000::/10"},
			dualStack: false,
		},
		{
			desc:      "false despite dual-stack because 3rd cidr is invalid",
			cidrs:     []string{"10.0.0.0/8", "2000::/10", "not-a-valid-cidr"},
			dualStack: false,
		},
		{
			desc:      "valid dual-stack",
			cidrs:     []string{"10.0.0.0/8", "2000::/10"},
			dualStack: true,
		},
		{
			desc:      "valid dual-stack, ipv6-primary",
			cidrs:     []string{"2000::/10", "10.0.0.0/8"},
			dualStack: true,
		},
		{
			desc:      "valid dual-stack, multiple ipv6",
			cidrs:     []string{"2000::/10", "10.0.0.0/8", "3000::/10"},
			dualStack: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			ipnets := make([]*net.IPNet, len(tc.cidrs))
			prefixes := make([]netip.Prefix, len(tc.cidrs))
			for i := range tc.cidrs {
				_, ipnets[i], _ = netutils.ParseCIDRSloppy(tc.cidrs[i])
				prefixes[i], _ = netip.ParsePrefix(tc.cidrs[i])
			}

			dualStack := IsDualStackCIDRs(tc.cidrs)
			if dualStack != tc.dualStack {
				t.Errorf("expected %v []string got %v", tc.dualStack, dualStack)
			}
			if IsDualStackCIDRPair(tc.cidrs) != (dualStack && len(tc.cidrs) == 2) {
				t.Errorf("IsDualStackCIDRPair gave wrong result for []string")
			}

			dualStack = IsDualStackCIDRs(ipnets)
			if dualStack != tc.dualStack {
				t.Errorf("expected %v []*net.IPNet got %v", tc.dualStack, dualStack)
			}
			if IsDualStackCIDRPair(ipnets) != (dualStack && len(tc.cidrs) == 2) {
				t.Errorf("IsDualStackCIDRPair gave wrong result for []*net.IPNet")
			}

			dualStack = IsDualStackCIDRs(prefixes)
			if dualStack != tc.dualStack {
				t.Errorf("expected %v []netip.Prefix got %v", tc.dualStack, dualStack)
			}
			if IsDualStackCIDRPair(prefixes) != (dualStack && len(tc.cidrs) == 2) {
				t.Errorf("IsDualStackCIDRPair gave wrong result for []netip.Prefix")
			}
		})
	}
}
