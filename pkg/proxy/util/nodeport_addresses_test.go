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
	"fmt"
	"net"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	fake "k8s.io/kubernetes/pkg/proxy/util/testing"
	netutils "k8s.io/utils/net"
)

type InterfaceAddrsPair struct {
	itf   net.Interface
	addrs []net.Addr
}

func TestGetNodeAddresses(t *testing.T) {
	testCases := []struct {
		cidrs         []string
		nw            *fake.FakeNetwork
		itfAddrsPairs []InterfaceAddrsPair
		expected      sets.String
		expectedErr   error
	}{
		{ // case 0
			cidrs: []string{"10.20.30.0/24"},
			nw:    fake.NewFakeNetwork(),
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
			expected: sets.NewString("10.20.30.51"),
		},
		{ // case 1
			cidrs: []string{"0.0.0.0/0"},
			nw:    fake.NewFakeNetwork(),
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
			expected: sets.NewString("0.0.0.0/0"),
		},
		{ // case 2
			cidrs: []string{"2001:db8::/32", "::1/128"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("2001:db8::1"), Mask: net.CIDRMask(32, 128)}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("::1"), Mask: net.CIDRMask(128, 128)}},
				},
			},
			expected: sets.NewString("2001:db8::1", "::1"),
		},
		{ // case 3
			cidrs: []string{"::/0"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("2001:db8::1"), Mask: net.CIDRMask(32, 128)}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("::1"), Mask: net.CIDRMask(128, 128)}},
				},
			},
			expected: sets.NewString("::/0"),
		},
		{ // case 4
			cidrs: []string{"127.0.0.1/32"},
			nw:    fake.NewFakeNetwork(),
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
			expected: sets.NewString("127.0.0.1"),
		},
		{ // case 5
			cidrs: []string{"127.0.0.0/8"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("127.0.1.1"), Mask: net.CIDRMask(8, 32)}},
				},
			},
			expected: sets.NewString("127.0.1.1"),
		},
		{ // case 6
			cidrs: []string{"10.20.30.0/24", "100.200.201.0/24"},
			nw:    fake.NewFakeNetwork(),
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
			expected: sets.NewString("10.20.30.51", "100.200.201.1"),
		},
		{ // case 7
			cidrs: []string{"10.20.30.0/24", "100.200.201.0/24"},
			nw:    fake.NewFakeNetwork(),
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
			expected:    nil,
			expectedErr: fmt.Errorf("no addresses found for cidrs %v", []string{"10.20.30.0/24", "100.200.201.0/24"}),
		},
		{ // case 8
			cidrs: []string{},
			nw:    fake.NewFakeNetwork(),
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
			expected: sets.NewString("0.0.0.0/0", "::/0"),
		},
		{ // case 9
			cidrs: []string{},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("2001:db8::1"), Mask: net.CIDRMask(32, 128)}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("::1"), Mask: net.CIDRMask(128, 128)}},
				},
			},
			expected: sets.NewString("0.0.0.0/0", "::/0"),
		},
		{ // case 9
			cidrs: []string{"1.2.3.0/24", "0.0.0.0/0"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("1.2.3.4"), Mask: net.CIDRMask(30, 32)}},
				},
			},
			expected: sets.NewString("0.0.0.0/0"),
		},
		{ // case 10
			cidrs: []string{"0.0.0.0/0", "1.2.3.0/24", "::1/128"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("1.2.3.4"), Mask: net.CIDRMask(30, 32)}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("::1"), Mask: net.CIDRMask(128, 128)}},
				},
			},
			expected: sets.NewString("0.0.0.0/0", "::1"),
		},
		{ // case 11
			cidrs: []string{"::/0", "1.2.3.0/24", "::1/128"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("1.2.3.4"), Mask: net.CIDRMask(30, 32)}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{&net.IPNet{IP: netutils.ParseIPSloppy("::1"), Mask: net.CIDRMask(128, 128)}},
				},
			},
			expected: sets.NewString("::/0", "1.2.3.4"),
		},
	}

	for i := range testCases {
		for _, pair := range testCases[i].itfAddrsPairs {
			testCases[i].nw.AddInterfaceAddr(&pair.itf, pair.addrs)
		}
		addrList, err := GetNodeAddresses(testCases[i].cidrs, testCases[i].nw)
		if !reflect.DeepEqual(err, testCases[i].expectedErr) {
			t.Errorf("case [%d], unexpected error: %v", i, err)
		}

		if !addrList.Equal(testCases[i].expected) {
			t.Errorf("case [%d], unexpected mismatch, expected: %v, got: %v", i, testCases[i].expected, addrList)
		}
	}
}

func TestContainsIPv4Loopback(t *testing.T) {
	tests := []struct {
		name        string
		cidrStrings []string
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
			name:        "all zeros ipv4 and invalid cidr",
			cidrStrings: []string{"invalid.cidr", "192.168.0.0/16", "fd00:1:d::/64", "0.0.0.0/0"},
			want:        true,
		},
		{
			name:        "all zeros ipv6", // interpret all zeros equal for IPv4 and IPv6 as Golang stdlib
			cidrStrings: []string{"224.0.0.0/24", "192.168.0.0/16", "fd00:1:d::/64", "::/0"},
			want:        true,
		},
		{
			name:        "ipv4 loopback",
			cidrStrings: []string{"224.0.0.0/24", "192.168.0.0/16", "fd00:1:d::/64", "127.0.0.0/8"},
			want:        true,
		},
		{
			name:        "ipv6 loopback",
			cidrStrings: []string{"224.0.0.0/24", "192.168.0.0/16", "fd00:1:d::/64", "::1/128"},
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
		{
			name:        "invalid cidr",
			cidrStrings: []string{"invalid.ip/invalid.mask"},
			want:        false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ContainsIPv4Loopback(tt.cidrStrings); got != tt.want {
				t.Errorf("ContainLoopback() = %v, want %v", got, tt.want)
			}
		})
	}
}
