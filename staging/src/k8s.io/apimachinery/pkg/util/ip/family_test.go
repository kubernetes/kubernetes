/*
Copyright 2023 The Kubernetes Authors.

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
	"net/netip"
	"testing"

	"k8s.io/api/core/v1"
)

func TestIPFamilyOfString(t *testing.T) {
	testCases := []struct {
		desc   string
		ip     string
		family v1.IPFamily
	}{
		{
			desc:   "IPv4 1",
			ip:     "127.0.0.1",
			family: v1.IPv4Protocol,
		},
		{
			desc:   "IPv4 2",
			ip:     "192.168.0.0",
			family: v1.IPv4Protocol,
		},
		{
			desc:   "IPv4 3",
			ip:     "1.2.3.4",
			family: v1.IPv4Protocol,
		},
		{
			desc:   "IPv4 with leading 0s is accepted",
			ip:     "001.002.003.004",
			family: v1.IPv4Protocol,
		},
		{
			desc:   "IPv4 encoded as IPv6 is IPv4",
			ip:     "::FFFF:1.2.3.4",
			family: v1.IPv4Protocol,
		},
		{
			desc:   "IPv6 1",
			ip:     "::1",
			family: v1.IPv6Protocol,
		},
		{
			desc:   "IPv6 2",
			ip:     "fd00::600d:f00d",
			family: v1.IPv6Protocol,
		},
		{
			desc:   "IPv6 3",
			ip:     "2001:db8::5",
			family: v1.IPv6Protocol,
		},
		{
			desc:   "IPv4 with out-of-range octets is not accepted",
			ip:     "1.2.3.400",
			family: "",
		},
		{
			desc:   "IPv6 with out-of-range segment is not accepted",
			ip:     "2001:db8::10005",
			family: "",
		},
		{
			desc:   "IPv4 with empty octet is not accepted",
			ip:     "1.2..4",
			family: "",
		},
		{
			desc:   "IPv6 with multiple empty segments is not accepted",
			ip:     "2001::db8::5",
			family: "",
		},
		{
			desc:   "IPv4 CIDR is not accepted",
			ip:     "1.2.3.4/32",
			family: "",
		},
		{
			desc:   "IPv6 CIDR is not accepted",
			ip:     "2001:db8::/64",
			family: "",
		},
		{
			desc:   "IPv4:port is not accepted",
			ip:     "1.2.3.4:80",
			family: "",
		},
		{
			desc:   "[IPv6] with brackets is not accepted",
			ip:     "[2001:db8::5]",
			family: "",
		},
		{
			desc:   "[IPv6]:port is not accepted",
			ip:     "[2001:db8::5]:80",
			family: "",
		},
		{
			desc:   "IPv6%zone is not accepted",
			ip:     "fe80::1234%eth0",
			family: "",
		},
		{
			desc:   "IPv4 with leading whitespace is not accepted",
			ip:     " 1.2.3.4",
			family: "",
		},
		{
			desc:   "IPv4 with trailing whitespace is not accepted",
			ip:     "1.2.3.4 ",
			family: "",
		},
		{
			desc:   "IPv6 with leading whitespace is not accepted",
			ip:     " 2001:db8::5",
			family: "",
		},
		{
			desc:   "IPv6 with trailing whitespace is not accepted",
			ip:     " 2001:db8::5",
			family: "",
		},
		{
			desc:   "random unparseable string",
			ip:     "bad ip",
			family: "",
		},
	}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			family := IPFamilyOfString(tc.ip)
			isIPv4 := IsIPv4String(tc.ip)
			isIPv6 := IsIPv6String(tc.ip)

			if family != tc.family {
				t.Errorf("Expect %q family %q, got %q", tc.ip, tc.family, family)
			}
			if isIPv4 != (tc.family == v1.IPv4Protocol) {
				t.Errorf("Expect %q ipv4 %v, got %v", tc.ip, tc.family == v1.IPv4Protocol, isIPv6)
			}
			if isIPv6 != (tc.family == v1.IPv6Protocol) {
				t.Errorf("Expect %q ipv6 %v, got %v", tc.ip, tc.family == v1.IPv6Protocol, isIPv6)
			}
		})
	}
}

func TestIsIPFamilyOf(t *testing.T) {
	testCases := []struct {
		desc   string
		ip     netip.Addr
		family v1.IPFamily
	}{
		{
			desc:   "IPv4 all-zeros",
			ip:     netip.IPv4Unspecified(),
			family: v1.IPv4Protocol,
		},
		{
			desc:   "IPv6 all-zeros",
			ip:     netip.IPv6Unspecified(),
			family: v1.IPv6Protocol,
		},
		{
			desc:   "IPv4 broadcast",
			ip:     netip.AddrFrom4([4]byte{0xFF, 0xFF, 0xFF, 0xFF}),
			family: v1.IPv4Protocol,
		},
		{
			desc:   "IPv4 loopback",
			ip:     netip.AddrFrom4([4]byte{127, 0, 0, 1}),
			family: v1.IPv4Protocol,
		},
		{
			desc:   "IPv6 loopback",
			ip:     netip.IPv6Loopback(),
			family: v1.IPv6Protocol,
		},
		{
			desc:   "IPv4 1",
			ip:     netip.MustParseAddr("10.20.40.40"),
			family: v1.IPv4Protocol,
		},
		{
			desc:   "IPv4 2",
			ip:     netip.MustParseAddr("172.17.3.0"),
			family: v1.IPv4Protocol,
		},
		{
			desc:   "IPv4 encoded as IPv6 is IPv4",
			ip:     netip.MustParseAddr("::FFFF:1.2.3.4"),
			family: v1.IPv4Protocol,
		},
		{
			desc:   "IPv6 1",
			ip:     netip.MustParseAddr("fd00::600d:f00d"),
			family: v1.IPv6Protocol,
		},
		{
			desc:   "IPv6 2",
			ip:     netip.MustParseAddr("2001:db8::5"),
			family: v1.IPv6Protocol,
		},
		{
			desc:   "zero IP is accepted, but is neither IPv4 nor IPv6",
			ip:     netip.Addr{},
			family: "",
		},
	}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			family := IPFamilyOf(tc.ip)
			isIPv4 := IsIPv4(tc.ip)
			isIPv6 := IsIPv6(tc.ip)

			if family != tc.family {
				t.Errorf("Expect family %q, got %q", tc.family, family)
			}
			if isIPv4 != (tc.family == v1.IPv4Protocol) {
				t.Errorf("Expect ipv4 %v, got %v", tc.family == v1.IPv4Protocol, isIPv6)
			}
			if isIPv6 != (tc.family == v1.IPv6Protocol) {
				t.Errorf("Expect ipv6 %v, got %v", tc.family == v1.IPv6Protocol, isIPv6)
			}
		})
	}
}

func TestIPFamilyOfCIDR(t *testing.T) {
	testCases := []struct {
		desc   string
		cidr   string
		family v1.IPFamily
	}{
		{
			desc:   "IPv4 CIDR 1",
			cidr:   "10.0.0.0/8",
			family: v1.IPv4Protocol,
		},
		{
			desc:   "IPv4 CIDR 2",
			cidr:   "192.168.0.0/16",
			family: v1.IPv4Protocol,
		},
		{
			desc:   "IPv6 CIDR 1",
			cidr:   "::/1",
			family: v1.IPv6Protocol,
		},
		{
			desc:   "IPv6 CIDR 2",
			cidr:   "2000::/10",
			family: v1.IPv6Protocol,
		},
		{
			desc:   "IPv6 CIDR 3",
			cidr:   "2001:db8::/32",
			family: v1.IPv6Protocol,
		},
		{
			desc:   "bad CIDR",
			cidr:   "foo",
			family: "",
		},
		{
			desc:   "IPv4 with out-of-range octets is not accepted",
			cidr:   "1.2.3.400/32",
			family: "",
		},
		{
			desc:   "IPv6 with out-of-range segment is not accepted",
			cidr:   "2001:db8::10005/64",
			family: "",
		},
		{
			desc:   "IPv4 with out-of-range mask length is not accepted",
			cidr:   "1.2.3.4/64",
			family: "",
		},
		{
			desc:   "IPv6 with out-of-range mask length is not accepted",
			cidr:   "2001:db8::5/192",
			family: "",
		},
		{
			desc:   "IPv4 with empty octet is not accepted",
			cidr:   "1.2..4/32",
			family: "",
		},
		{
			desc:   "IPv6 with multiple empty segments is not accepted",
			cidr:   "2001::db8::5/64",
			family: "",
		},
		{
			desc:   "IPv4 IP is not CIDR",
			cidr:   "192.168.0.0",
			family: "",
		},
		{
			desc:   "IPv6 IP is not CIDR",
			cidr:   "2001:db8::",
			family: "",
		},
		{
			desc:   "IPv4 CIDR with leading whitespace is not accepted",
			cidr:   " 1.2.3.4/32",
			family: "",
		},
		{
			desc:   "IPv4 CIDR with trailing whitespace is not accepted",
			cidr:   "1.2.3.4/32 ",
			family: "",
		},
		{
			desc:   "IPv6 CIDR with leading whitespace is not accepted",
			cidr:   " 2001:db8::5/64",
			family: "",
		},
		{
			desc:   "IPv6 CIDR with trailing whitespace is not accepted",
			cidr:   " 2001:db8::5/64",
			family: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			family := IPFamilyOfCIDRString(tc.cidr)
			isIPv4 := IsIPv4CIDRString(tc.cidr)
			isIPv6 := IsIPv6CIDRString(tc.cidr)

			if family != tc.family {
				t.Errorf("Expect family %v, got %v", tc.family, family)
			}
			if isIPv4 != (tc.family == v1.IPv4Protocol) {
				t.Errorf("Expect %q ipv4 %v, got %v", tc.cidr, tc.family == v1.IPv4Protocol, isIPv6)
			}
			if isIPv6 != (tc.family == v1.IPv6Protocol) {
				t.Errorf("Expect %q ipv6 %v, got %v", tc.cidr, tc.family == v1.IPv6Protocol, isIPv6)
			}

			parsed, err := netip.ParsePrefix(tc.cidr)
			if err != nil {
				if tc.family != "" {
					t.Errorf("unexpected error parsing %q", tc.cidr)
				}
				return
			}

			familyParsed := IPFamilyOfCIDR(parsed)
			isIPv4Parsed := IsIPv4CIDR(parsed)
			isIPv6Parsed := IsIPv6CIDR(parsed)
			if familyParsed != family {
				t.Errorf("%q gives different results for IPFamilyOfCIDR (%v) and IPFamilyOfCIDRString (%v)", tc.cidr, familyParsed, family)
			}
			if isIPv4Parsed != isIPv4 {
				t.Errorf("%q gives different results for IsIPv4CIDR (%v) and IsIPv4CIDRString (%v)", tc.cidr, isIPv4Parsed, isIPv4)
			}
			if isIPv6Parsed != isIPv6 {
				t.Errorf("%q gives different results for IsIPv6CIDR (%v) and IsIPv6CIDRString (%v)", tc.cidr, isIPv6Parsed, isIPv6)
			}
		})
	}
}

func TestDualStackIPs(t *testing.T) {
	testCases := []struct {
		desc           string
		ips            []string
		expectedResult bool
		expectError    bool
	}{
		{
			desc:           "false because length is not at least 2",
			ips:            []string{"1.1.1.1"},
			expectedResult: false,
			expectError:    false,
		},
		{
			desc:           "false because length is not at least 2",
			ips:            []string{},
			expectedResult: false,
			expectError:    false,
		},
		{
			desc:           "false because all are v4",
			ips:            []string{"1.1.1.1", "2.2.2.2", "3.3.3.3"},
			expectedResult: false,
			expectError:    false,
		},
		{
			desc:           "false because all are v6",
			ips:            []string{"fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff", "fd92:20ba:ca:34f7:ffff:ffff:ffff:fff0", "fd92:20ba:ca:34f7:ffff:ffff:ffff:fff1"},
			expectedResult: false,
			expectError:    false,
		},
		{
			desc:           "false because 2nd ip is invalid",
			ips:            []string{"1.1.1.1", "not-a-valid-ip"},
			expectedResult: false,
			expectError:    true,
		},
		{
			desc:           "false because 1st ip is invalid",
			ips:            []string{"not-a-valid-ip", "fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff"},
			expectedResult: false,
			expectError:    true,
		},
		{
			desc:           "false despite dual-stack IPs because 3rd IP is invalid",
			ips:            []string{"1.1.1.1", "fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff", "not-a-valid-ip"},
			expectedResult: false,
			expectError:    true,
		},
		{
			desc:           "valid dual-stack",
			ips:            []string{"1.1.1.1", "fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff"},
			expectedResult: true,
			expectError:    false,
		},
		{
			desc:           "valid dual stack with multiple IPv6",
			ips:            []string{"fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff", "1.1.1.1", "fd92:20ba:ca:34f7:ffff:ffff:ffff:fff0"},
			expectedResult: true,
			expectError:    false,
		},
		{
			desc:           "valid dual stack with multiple IPv4",
			ips:            []string{"1.1.1.1", "fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff", "10.0.0.0"},
			expectedResult: true,
			expectError:    false,
		},
		{
			desc:           "valid dual-stack, IPv6-primary",
			ips:            []string{"fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff", "1.1.1.1"},
			expectedResult: true,
			expectError:    false,
		},
	}
	// for each test case, test the regular func and the string func
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			dualStack := IsDualStackIPStrings(tc.ips)
			if dualStack != tc.expectedResult {
				t.Errorf("expected %v got %v", tc.expectedResult, dualStack)
			}
		})
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			var err error
			ips := make([]netip.Addr, len(tc.ips))
			for i := range tc.ips {
				ips[i], err = netip.ParseAddr(tc.ips[i])
				if err != nil {
					if !tc.expectError {
						t.Errorf("failed to parse expected-valid IP %q: %v", tc.ips[i], err)
					}
					return
				}
			}
			if tc.expectError {
				t.Errorf("expected a parse error on %v but didn't get one", tc.ips)
			}

			dualStack := IsDualStackIPs(ips)
			if dualStack != tc.expectedResult {
				t.Errorf("expected %v got %v", tc.expectedResult, dualStack)
			}
		})
	}
}

func TestDualStackCIDRs(t *testing.T) {
	testCases := []struct {
		desc           string
		cidrs          []string
		expectedResult bool
		expectError    bool
	}{
		{
			desc:           "false because length is not at least 2",
			cidrs:          []string{"10.10.10.10/8"},
			expectedResult: false,
			expectError:    false,
		},
		{
			desc:           "false because length is not at least 2",
			cidrs:          []string{},
			expectedResult: false,
			expectError:    false,
		},
		{
			desc:           "false because all cidrs are v4",
			cidrs:          []string{"10.10.10.10/8", "20.20.20.20/8", "30.30.30.30/8"},
			expectedResult: false,
			expectError:    false,
		},
		{
			desc:           "false because all cidrs are v6",
			cidrs:          []string{"2000::/10", "3000::/10"},
			expectedResult: false,
			expectError:    false,
		},
		{
			desc:           "false because 2nd cidr is invalid",
			cidrs:          []string{"10.10.10.10/8", "not-a-valid-cidr"},
			expectedResult: false,
			expectError:    true,
		},
		{
			desc:           "false because 1st cidr is invalid",
			cidrs:          []string{"not-a-valid-ip", "2000::/10"},
			expectedResult: false,
			expectError:    true,
		},
		{
			desc:           "false despite dual-stack because 3rd cidr is invalid",
			cidrs:          []string{"10.10.10.10/8", "2000::/10", "not-a-valid-cidr"},
			expectedResult: false,
			expectError:    true,
		},
		{
			desc:           "valid dual-stack",
			cidrs:          []string{"10.10.10.10/8", "2000::/10"},
			expectedResult: true,
			expectError:    false,
		},
		{
			desc:           "valid dual-stack, ipv6-primary",
			cidrs:          []string{"2000::/10", "10.10.10.10/8"},
			expectedResult: true,
			expectError:    false,
		},
		{
			desc:           "valid dual-stack, multiple ipv6",
			cidrs:          []string{"2000::/10", "10.10.10.10/8", "3000::/10"},
			expectedResult: true,
			expectError:    false,
		},
	}

	// for each test case, test the regular func and the string func
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			dualStack := IsDualStackCIDRStrings(tc.cidrs)
			if dualStack != tc.expectedResult {
				t.Errorf("expected %v got %v", tc.expectedResult, dualStack)
			}
		})
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			var err error
			cidrs := make([]netip.Prefix, len(tc.cidrs))
			for i := range tc.cidrs {
				cidrs[i], err = netip.ParsePrefix(tc.cidrs[i])
				if err != nil {
					if !tc.expectError {
						t.Errorf("failed to parse expected-valid CIDR %q: %v", tc.cidrs[i], err)
					}
					return
				}
			}
			if tc.expectError {
				t.Errorf("expected a parse error on %v but didn't get one", tc.cidrs)
			}

			dualStack := IsDualStackCIDRs(cidrs)
			if dualStack != tc.expectedResult {
				t.Errorf("expected %v got %v", tc.expectedResult, dualStack)
			}
		})
	}
}
