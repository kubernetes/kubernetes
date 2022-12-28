/*
Copyright 2022 The Kubernetes Authors.

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

package util

import (
	"net"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	fake "k8s.io/kubernetes/pkg/proxy/util/testing"
	netutils "k8s.io/utils/net"
)

type InterfaceAddrsPair struct {
	itf   net.Interface
	addrs []net.Addr
}

func TestGetNodeAddresses(t *testing.T) {
	type expectation struct {
		matchAll bool
		ips      sets.String
		lbIPs    sets.String
	}

	testCases := []struct {
		name          string
		cidrs         []string
		itfAddrsPairs []InterfaceAddrsPair
		expected      map[v1.IPFamily]expectation
	}{
		{
			name:  "IPv4 single",
			cidrs: []string{"10.20.30.0/24"},
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("10.20.30.51"), Mask: net.CIDRMask(24, 32)}},
				},
				{
					itf:   net.Interface{Index: 2, MTU: 0, Name: "eth1", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("100.200.201.1"), Mask: net.CIDRMask(24, 32)}},
				},
			},
			expected: map[v1.IPFamily]expectation{
				v1.IPv4Protocol: {
					ips: sets.NewString("10.20.30.51"),
				},
				v1.IPv6Protocol: {
					matchAll: true,
					ips:      nil,
				},
			},
		},
		{
			name:  "IPv4 zero CIDR",
			cidrs: []string{"0.0.0.0/0"},
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("10.20.30.51"), Mask: net.CIDRMask(24, 32)}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("127.0.0.1"), Mask: net.CIDRMask(8, 32)}},
				},
			},
			expected: map[v1.IPFamily]expectation{
				v1.IPv4Protocol: {
					matchAll: true,
					ips:      sets.NewString("10.20.30.51"),
					lbIPs:    sets.NewString("127.0.0.1"),
				},
				v1.IPv6Protocol: {
					matchAll: true,
					ips:      nil,
				},
			},
		},
		{
			name:  "IPv6 multiple",
			cidrs: []string{"2001:db8::/64", "::1/128"},
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("2001:db8::1"), Mask: net.CIDRMask(64, 128)}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("::1"), Mask: net.CIDRMask(128, 128)}},
				},
			},
			expected: map[v1.IPFamily]expectation{
				v1.IPv4Protocol: {
					matchAll: true,
					ips:      nil,
				},
				v1.IPv6Protocol: {
					ips: sets.NewString("2001:db8::1"),
				},
			},
		},
		{
			name:  "IPv6 zero CIDR",
			cidrs: []string{"::/0"},
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("2001:db8::1"), Mask: net.CIDRMask(64, 128)}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("::1"), Mask: net.CIDRMask(128, 128)}},
				},
			},
			expected: map[v1.IPFamily]expectation{
				v1.IPv4Protocol: {
					matchAll: true,
					ips:      nil,
				},
				v1.IPv6Protocol: {
					matchAll: true,
					ips:      sets.NewString("2001:db8::1"),
				},
			},
		},
		{
			name:  "IPv4 localhost exact",
			cidrs: []string{"127.0.0.1/32"},
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("10.20.30.51"), Mask: net.CIDRMask(24, 32)}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("127.0.0.1"), Mask: net.CIDRMask(8, 32)}},
				},
			},
			expected: map[v1.IPFamily]expectation{
				v1.IPv4Protocol: {
					ips:   nil,
					lbIPs: sets.NewString("127.0.0.1"),
				},
				v1.IPv6Protocol: {
					matchAll: true,
					ips:      nil,
				},
			},
		},
		{
			name:  "IPv4 localhost subnet",
			cidrs: []string{"127.0.0.0/8"},
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("127.0.1.1"), Mask: net.CIDRMask(8, 32)}},
				},
			},
			expected: map[v1.IPFamily]expectation{
				v1.IPv4Protocol: {
					ips:   nil,
					lbIPs: sets.NewString("127.0.1.1"),
				},
				v1.IPv6Protocol: {
					matchAll: true,
					ips:      nil,
				},
			},
		},
		{
			name:  "IPv4 multiple",
			cidrs: []string{"10.20.30.0/24", "100.200.201.0/24"},
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("10.20.30.51"), Mask: net.CIDRMask(24, 32)}},
				},
				{
					itf:   net.Interface{Index: 2, MTU: 0, Name: "eth1", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("100.200.201.1"), Mask: net.CIDRMask(24, 32)}},
				},
			},
			expected: map[v1.IPFamily]expectation{
				v1.IPv4Protocol: {
					ips: sets.NewString("10.20.30.51", "100.200.201.1"),
				},
				v1.IPv6Protocol: {
					matchAll: true,
					ips:      nil,
				},
			},
		},
		{
			name:  "IPv4 multiple, no match",
			cidrs: []string{"10.20.30.0/24", "100.200.201.0/24"},
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("192.168.1.2"), Mask: net.CIDRMask(24, 32)}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("127.0.0.1"), Mask: net.CIDRMask(8, 32)}},
				},
			},
			expected: map[v1.IPFamily]expectation{
				v1.IPv4Protocol: {
					ips: nil,
				},
				v1.IPv6Protocol: {
					matchAll: true,
					ips:      nil,
				},
			},
		},
		{
			name:  "empty list, IPv4 addrs",
			cidrs: []string{},
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("192.168.1.2"), Mask: net.CIDRMask(24, 32)}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("127.0.0.1"), Mask: net.CIDRMask(8, 32)}},
				},
			},
			expected: map[v1.IPFamily]expectation{
				v1.IPv4Protocol: {
					matchAll: true,
					ips:      sets.NewString("192.168.1.2"),
					lbIPs:    sets.NewString("127.0.0.1"),
				},
				v1.IPv6Protocol: {
					matchAll: true,
					ips:      nil,
				},
			},
		},
		{
			name:  "empty list, IPv6 addrs",
			cidrs: []string{},
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("2001:db8::1"), Mask: net.CIDRMask(64, 128)}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("::1"), Mask: net.CIDRMask(128, 128)}},
				},
			},
			expected: map[v1.IPFamily]expectation{
				v1.IPv4Protocol: {
					matchAll: true,
					ips:      nil,
				},
				v1.IPv6Protocol: {
					matchAll: true,
					ips:      sets.NewString("2001:db8::1"),
				},
			},
		},
		{
			name:  "IPv4 redundant CIDRs",
			cidrs: []string{"1.2.3.0/24", "0.0.0.0/0"},
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("1.2.3.4"), Mask: net.CIDRMask(30, 32)}},
				},
			},
			expected: map[v1.IPFamily]expectation{
				v1.IPv4Protocol: {
					matchAll: true,
					ips:      sets.NewString("1.2.3.4"),
				},
				v1.IPv6Protocol: {
					matchAll: true,
					ips:      nil,
				},
			},
		},
		{
			name:  "Dual-stack, redundant IPv4",
			cidrs: []string{"0.0.0.0/0", "1.2.3.0/24", "2001:db8::1/128"},
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf: net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{
						&net.IPNet{IP: netutils.ParseIPSloppy("1.2.3.4"), Mask: net.CIDRMask(30, 32)},
						&net.IPNet{IP: netutils.ParseIPSloppy("2001:db8::1"), Mask: net.CIDRMask(64, 128)},
					},
				},
				{
					itf: net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{
						&net.IPNet{IP: netutils.ParseIPSloppy("127.0.0.1"), Mask: net.CIDRMask(8, 32)},
						&net.IPNet{IP: netutils.ParseIPSloppy("::1"), Mask: net.CIDRMask(128, 128)},
					},
				},
			},
			expected: map[v1.IPFamily]expectation{
				v1.IPv4Protocol: {
					matchAll: true,
					ips:      sets.NewString("1.2.3.4"),
					lbIPs:    sets.NewString("127.0.0.1"),
				},
				v1.IPv6Protocol: {
					ips: sets.NewString("2001:db8::1"),
				},
			},
		},
		{
			name:  "Dual-stack, redundant IPv6",
			cidrs: []string{"::/0", "1.2.3.0/24", "2001:db8::1/128"},
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf: net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{
						&net.IPNet{IP: netutils.ParseIPSloppy("1.2.3.4"), Mask: net.CIDRMask(30, 32)},
						&net.IPNet{IP: netutils.ParseIPSloppy("2001:db8::1"), Mask: net.CIDRMask(64, 128)},
					},
				},
				{
					itf: net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{
						&net.IPNet{IP: netutils.ParseIPSloppy("127.0.0.1"), Mask: net.CIDRMask(8, 32)},
						&net.IPNet{IP: netutils.ParseIPSloppy("::1"), Mask: net.CIDRMask(128, 128)},
					},
				},
			},
			expected: map[v1.IPFamily]expectation{
				v1.IPv4Protocol: {
					ips: sets.NewString("1.2.3.4"),
				},
				v1.IPv6Protocol: {
					matchAll: true,
					ips:      sets.NewString("2001:db8::1"),
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			nw := fake.NewFakeNetwork()
			for _, pair := range tc.itfAddrsPairs {
				nw.AddInterfaceAddr(&pair.itf, pair.addrs)
			}

			for _, family := range []v1.IPFamily{v1.IPv4Protocol, v1.IPv6Protocol} {
				npa := NewNodePortAddresses(family, tc.cidrs, false)

				if npa.MatchAll() != tc.expected[family].matchAll {
					t.Errorf("unexpected MatchAll(%s), expected: %v", family, tc.expected[family].matchAll)
				}

				addrList, err := npa.GetNodeAddresses(nw)
				expectedIPs := tc.expected[family].ips

				// The fake InterfaceAddrs() never returns an error, so
				// the only error GetNodeAddresses will return is "no
				// addresses found".
				if err != nil && len(expectedIPs) != 0 {
					t.Errorf("unexpected error for %s: %v", family, err)
				}
				if !sets.NewString(addrList...).Equal(expectedIPs) {
					t.Errorf("unexpected mismatch for %s, expected: %v, got: %v", family, expectedIPs, addrList)
				}
			}

			// Now with allowIPv4Localhost=true
			npa := NewNodePortAddresses(v1.IPv4Protocol, tc.cidrs, true)
			addrList, err := npa.GetNodeAddresses(nw)
			expectedIPs := tc.expected[v1.IPv4Protocol].ips.Union(tc.expected[v1.IPv4Protocol].lbIPs)
			if err != nil && len(expectedIPs) != 0 {
				t.Errorf("unexpected error for ipv4Localhost=true: %v", err)
			}
			if !sets.NewString(addrList...).Equal(expectedIPs) {
				t.Errorf("unexpected mismatch with ipv4Localhost=true, expected: %v, got: %v", expectedIPs, addrList)
			}
		})
	}
}

func TestAllowLocalhost(t *testing.T) {
	tests := []struct {
		name        string
		cidrStrings []string
		disallow    bool
		want        bool
	}{
		{
			name: "empty",
			want: true,
		},
		{
			name:        "all zeros ipv4",
			cidrStrings: []string{"224.0.0.0/24", "192.168.0.0/16", "fd00:1:d::/64", "0.0.0.0/0"},
			want:        true,
		},
		{
			name:        "all zeros ipv6",
			cidrStrings: []string{"224.0.0.0/24", "192.168.0.0/16", "fd00:1:d::/64", "::/0"},
			want:        false,
		},
		{
			name:        "all zeros ipv4 but disallowed",
			cidrStrings: []string{"224.0.0.0/24", "192.168.0.0/16", "fd00:1:d::/64", "0.0.0.0/0"},
			disallow:    true,
			want:        false,
		},
		{
			name:        "all zeros ipv6 but disallowed",
			cidrStrings: []string{"224.0.0.0/24", "192.168.0.0/16", "fd00:1:d::/64", "::/0"},
			disallow:    true,
			want:        false,
		},
		{
			name:        "ipv4 loopback",
			cidrStrings: []string{"224.0.0.0/24", "192.168.0.0/16", "fd00:1:d::/64", "127.0.0.0/8"},
			want:        true,
		},
		{
			name:        "ipv4 loopback but disallowed",
			cidrStrings: []string{"224.0.0.0/24", "192.168.0.0/16", "fd00:1:d::/64", "127.0.0.0/8"},
			disallow:    true,
			want:        false,
		},
		{
			name:        "ipv6 loopback",
			cidrStrings: []string{"224.0.0.0/24", "192.168.0.0/16", "fd00:1:d::/64", "::1/128"},
			want:        false,
		},
		{
			name:        "ipv6 loopback but disallowed",
			cidrStrings: []string{"224.0.0.0/24", "192.168.0.0/16", "fd00:1:d::/64", "::1/128"},
			disallow:    true,
			want:        false,
		},
		{
			name:        "ipv4 loopback smaller range",
			cidrStrings: []string{"224.0.0.0/24", "192.168.0.0/16", "fd00:1:d::/64", "127.0.2.0/28"},
			want:        true,
		},
		{
			name:        "ipv4 loopback within larger range",
			cidrStrings: []string{"224.0.0.0/24", "192.168.0.0/16", "fd00:1:d::/64", "64.0.0.0/2"},
			want:        true,
		},
		{
			name:        "non loop loopback",
			cidrStrings: []string{"128.0.2.0/28", "224.0.0.0/24", "192.168.0.0/16", "fd00:1:d::/64"},
			want:        false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			npa := NewNodePortAddresses(v1.IPv4Protocol, tt.cidrStrings, !tt.disallow)
			if got := npa.AllowLocalhost(); got != tt.want {
				t.Errorf("IPv4 AllowLocalhost = %v, want %v", got, tt.want)
			}
			// AllowLocalhost should always be false for family=IPv6
			npa = NewNodePortAddresses(v1.IPv6Protocol, tt.cidrStrings, !tt.disallow)
			if npa.AllowLocalhost() {
				t.Errorf("IPv6 AllowLocalhost = true, want false")
			}
		})
	}
}
