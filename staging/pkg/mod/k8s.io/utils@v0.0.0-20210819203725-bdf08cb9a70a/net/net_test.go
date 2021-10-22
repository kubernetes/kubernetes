/*
Copyright 2018 The Kubernetes Authors.

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

package net

import (
	"net"
	"testing"
)

func TestParseCIDRs(t *testing.T) {
	testCases := []struct {
		cidrs         []string
		errString     string
		errorExpected bool
	}{
		{
			cidrs:         []string{},
			errString:     "should not return an error for an empty slice",
			errorExpected: false,
		},
		{
			cidrs:         []string{"10.0.0.0/8", "not-a-valid-cidr", "2000::/10"},
			errString:     "should return error for bad cidr",
			errorExpected: true,
		},
		{
			cidrs:         []string{"10.0.0.0/8", "2000::/10"},
			errString:     "should not return error for good  cidrs",
			errorExpected: false,
		},
	}

	for _, tc := range testCases {
		cidrs, err := ParseCIDRs(tc.cidrs)
		if tc.errorExpected {
			if err == nil {
				t.Errorf("%v", tc.errString)
			}
			continue
		}
		if err != nil {
			t.Errorf("%v error:%v", tc.errString, err)
		}

		// validate lengths
		if len(cidrs) != len(tc.cidrs) {
			t.Errorf("cidrs should be of the same lengths %v != %v", len(cidrs), len(tc.cidrs))
		}

	}
}
func TestDualStackIPs(t *testing.T) {
	testCases := []struct {
		ips            []string
		errMessage     string
		expectedResult bool
		expectError    bool
	}{
		{
			ips:            []string{"1.1.1.1"},
			errMessage:     "should fail because length is not at least 2",
			expectedResult: false,
			expectError:    false,
		},
		{
			ips:            []string{},
			errMessage:     "should fail because length is not at least 2",
			expectedResult: false,
			expectError:    false,
		},
		{
			ips:            []string{"1.1.1.1", "2.2.2.2", "3.3.3.3"},
			errMessage:     "should fail because all are v4",
			expectedResult: false,
			expectError:    false,
		},
		{
			ips:            []string{"fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff", "fd92:20ba:ca:34f7:ffff:ffff:ffff:fff0", "fd92:20ba:ca:34f7:ffff:ffff:ffff:fff1"},
			errMessage:     "should fail because all are v6",
			expectedResult: false,
			expectError:    false,
		},
		{
			ips:            []string{"1.1.1.1", "not-a-valid-ip"},
			errMessage:     "should fail because 2nd ip is invalid",
			expectedResult: false,
			expectError:    true,
		},
		{
			ips:            []string{"not-a-valid-ip", "fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff"},
			errMessage:     "should fail because 1st ip is invalid",
			expectedResult: false,
			expectError:    true,
		},
		{
			ips:            []string{"1.1.1.1", "fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff"},
			errMessage:     "expected success, but found failure",
			expectedResult: true,
			expectError:    false,
		},
		{
			ips:            []string{"fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff", "1.1.1.1", "fd92:20ba:ca:34f7:ffff:ffff:ffff:fff0"},
			errMessage:     "expected success, but found failure",
			expectedResult: true,
			expectError:    false,
		},
		{
			ips:            []string{"1.1.1.1", "fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff", "10.0.0.0"},
			errMessage:     "expected success, but found failure",
			expectedResult: true,
			expectError:    false,
		},
		{
			ips:            []string{"fd92:20ba:ca:34f7:ffff:ffff:ffff:ffff", "1.1.1.1"},
			errMessage:     "expected success, but found failure",
			expectedResult: true,
			expectError:    false,
		},
	}
	// for each test case, test the regular func and the string func
	for _, tc := range testCases {
		dualStack, err := IsDualStackIPStrings(tc.ips)
		if err == nil && tc.expectError {
			t.Errorf("%s", tc.errMessage)
			continue
		}
		if err != nil && !tc.expectError {
			t.Errorf("failed to run test case for %v, error: %v", tc.ips, err)
			continue
		}
		if dualStack != tc.expectedResult {
			t.Errorf("%v for %v", tc.errMessage, tc.ips)
		}
	}

	for _, tc := range testCases {
		ips := make([]net.IP, 0, len(tc.ips))
		for _, ip := range tc.ips {
			parsedIP := ParseIPSloppy(ip)
			ips = append(ips, parsedIP)
		}
		dualStack, err := IsDualStackIPs(ips)
		if err == nil && tc.expectError {
			t.Errorf("%s", tc.errMessage)
			continue
		}
		if err != nil && !tc.expectError {
			t.Errorf("failed to run test case for %v, error: %v", tc.ips, err)
			continue
		}
		if dualStack != tc.expectedResult {
			t.Errorf("%v for %v", tc.errMessage, tc.ips)
		}
	}
}

func TestDualStackCIDRs(t *testing.T) {
	testCases := []struct {
		cidrs          []string
		errMessage     string
		expectedResult bool
		expectError    bool
	}{
		{
			cidrs:          []string{"10.10.10.10/8"},
			errMessage:     "should fail because length is not at least 2",
			expectedResult: false,
			expectError:    false,
		},
		{
			cidrs:          []string{},
			errMessage:     "should fail because length is not at least 2",
			expectedResult: false,
			expectError:    false,
		},
		{
			cidrs:          []string{"10.10.10.10/8", "20.20.20.20/8", "30.30.30.30/8"},
			errMessage:     "should fail because all cidrs are v4",
			expectedResult: false,
			expectError:    false,
		},
		{
			cidrs:          []string{"2000::/10", "3000::/10"},
			errMessage:     "should fail because all cidrs are v6",
			expectedResult: false,
			expectError:    false,
		},
		{
			cidrs:          []string{"10.10.10.10/8", "not-a-valid-cidr"},
			errMessage:     "should fail because 2nd cidr is invalid",
			expectedResult: false,
			expectError:    true,
		},
		{
			cidrs:          []string{"not-a-valid-ip", "2000::/10"},
			errMessage:     "should fail because 1st cidr is invalid",
			expectedResult: false,
			expectError:    true,
		},
		{
			cidrs:          []string{"10.10.10.10/8", "2000::/10"},
			errMessage:     "expected success, but found failure",
			expectedResult: true,
			expectError:    false,
		},
		{
			cidrs:          []string{"2000::/10", "10.10.10.10/8"},
			errMessage:     "expected success, but found failure",
			expectedResult: true,
			expectError:    false,
		},
		{
			cidrs:          []string{"2000::/10", "10.10.10.10/8", "3000::/10"},
			errMessage:     "expected success, but found failure",
			expectedResult: true,
			expectError:    false,
		},
	}

	// for each test case, test the regular func and the string func
	for _, tc := range testCases {
		dualStack, err := IsDualStackCIDRStrings(tc.cidrs)
		if err == nil && tc.expectError {
			t.Errorf("%s", tc.errMessage)
			continue
		}
		if err != nil && !tc.expectError {
			t.Errorf("failed to run test case for %v, error: %v", tc.cidrs, err)
			continue
		}
		if dualStack != tc.expectedResult {
			t.Errorf("%v for %v", tc.errMessage, tc.cidrs)
		}
	}

	for _, tc := range testCases {
		cidrs := make([]*net.IPNet, 0, len(tc.cidrs))
		for _, cidr := range tc.cidrs {
			_, parsedCIDR, _ := ParseCIDRSloppy(cidr)
			cidrs = append(cidrs, parsedCIDR)
		}

		dualStack, err := IsDualStackCIDRs(cidrs)
		if err == nil && tc.expectError {
			t.Errorf("%s", tc.errMessage)
			continue
		}
		if err != nil && !tc.expectError {
			t.Errorf("failed to run test case for %v, error: %v", tc.cidrs, err)
			continue
		}
		if dualStack != tc.expectedResult {
			t.Errorf("%v for %v", tc.errMessage, tc.cidrs)
		}
	}
}

func TestIsIPv6String(t *testing.T) {
	testCases := []struct {
		ip         string
		expectIPv6 bool
	}{
		{
			ip:         "127.0.0.1",
			expectIPv6: false,
		},
		{
			ip:         "192.168.0.0",
			expectIPv6: false,
		},
		{
			ip:         "1.2.3.4",
			expectIPv6: false,
		},
		{
			ip:         "bad ip",
			expectIPv6: false,
		},
		{
			ip:         "::1",
			expectIPv6: true,
		},
		{
			ip:         "fd00::600d:f00d",
			expectIPv6: true,
		},
		{
			ip:         "2001:db8::5",
			expectIPv6: true,
		},
	}
	for i := range testCases {
		isIPv6 := IsIPv6String(testCases[i].ip)
		if isIPv6 != testCases[i].expectIPv6 {
			t.Errorf("[%d] Expect ipv6 %v, got %v", i+1, testCases[i].expectIPv6, isIPv6)
		}
	}
}

func TestIsIPv6(t *testing.T) {
	testCases := []struct {
		ip         net.IP
		expectIPv6 bool
	}{
		{
			ip:         net.IPv4zero,
			expectIPv6: false,
		},
		{
			ip:         net.IPv4bcast,
			expectIPv6: false,
		},
		{
			ip:         ParseIPSloppy("127.0.0.1"),
			expectIPv6: false,
		},
		{
			ip:         ParseIPSloppy("10.20.40.40"),
			expectIPv6: false,
		},
		{
			ip:         ParseIPSloppy("172.17.3.0"),
			expectIPv6: false,
		},
		{
			ip:         nil,
			expectIPv6: false,
		},
		{
			ip:         net.IPv6loopback,
			expectIPv6: true,
		},
		{
			ip:         net.IPv6zero,
			expectIPv6: true,
		},
		{
			ip:         ParseIPSloppy("fd00::600d:f00d"),
			expectIPv6: true,
		},
		{
			ip:         ParseIPSloppy("2001:db8::5"),
			expectIPv6: true,
		},
	}
	for i := range testCases {
		isIPv6 := IsIPv6(testCases[i].ip)
		if isIPv6 != testCases[i].expectIPv6 {
			t.Errorf("[%d] Expect ipv6 %v, got %v", i+1, testCases[i].expectIPv6, isIPv6)
		}
	}
}

func TestIsIPv6CIDRString(t *testing.T) {
	testCases := []struct {
		desc         string
		cidr         string
		expectResult bool
	}{
		{
			desc:         "ipv4 CIDR 1",
			cidr:         "10.0.0.0/8",
			expectResult: false,
		},
		{
			desc:         "ipv4 CIDR 2",
			cidr:         "192.168.0.0/16",
			expectResult: false,
		},
		{
			desc:         "ipv6 CIDR 1",
			cidr:         "::/1",
			expectResult: true,
		},
		{
			desc:         "ipv6 CIDR 2",
			cidr:         "2000::/10",
			expectResult: true,
		},
		{
			desc:         "ipv6 CIDR 3",
			cidr:         "2001:db8::/32",
			expectResult: true,
		},
	}

	for _, tc := range testCases {
		res := IsIPv6CIDRString(tc.cidr)
		if res != tc.expectResult {
			t.Errorf("%v: want IsIPv6CIDRString=%v, got %v", tc.desc, tc.expectResult, res)
		}
	}
}

func TestIsIPv6CIDR(t *testing.T) {
	testCases := []struct {
		desc         string
		cidr         string
		expectResult bool
	}{
		{
			desc:         "ipv4 CIDR 1",
			cidr:         "10.0.0.0/8",
			expectResult: false,
		},
		{
			desc:         "ipv4 CIDR 2",
			cidr:         "192.168.0.0/16",
			expectResult: false,
		},
		{
			desc:         "ipv6 CIDR 1",
			cidr:         "::/1",
			expectResult: true,
		},
		{
			desc:         "ipv6 CIDR 2",
			cidr:         "2000::/10",
			expectResult: true,
		},
		{
			desc:         "ipv6 CIDR 3",
			cidr:         "2001:db8::/32",
			expectResult: true,
		},
	}

	for _, tc := range testCases {
		_, cidr, _ := ParseCIDRSloppy(tc.cidr)
		res := IsIPv6CIDR(cidr)
		if res != tc.expectResult {
			t.Errorf("%v: want IsIPv6CIDR=%v, got %v", tc.desc, tc.expectResult, res)
		}
	}
}

func TestIsIPv4String(t *testing.T) {
	testCases := []struct {
		ip         string
		expectIPv4 bool
	}{
		{
			ip:         "127.0.0.1",
			expectIPv4: true,
		},
		{
			ip:         "192.168.0.0",
			expectIPv4: true,
		},
		{
			ip:         "1.2.3.4",
			expectIPv4: true,
		},
		{
			ip:         "bad ip",
			expectIPv4: false,
		},
		{
			ip:         "::1",
			expectIPv4: false,
		},
		{
			ip:         "fd00::600d:f00d",
			expectIPv4: false,
		},
		{
			ip:         "2001:db8::5",
			expectIPv4: false,
		},
	}
	for i := range testCases {
		isIPv4 := IsIPv4String(testCases[i].ip)
		if isIPv4 != testCases[i].expectIPv4 {
			t.Errorf("[%d] Expect ipv4 %v, got %v", i+1, testCases[i].expectIPv4, isIPv4)
		}
	}
}

func TestIsIPv4(t *testing.T) {
	testCases := []struct {
		ip         net.IP
		expectIPv4 bool
	}{
		{
			ip:         net.IPv4zero,
			expectIPv4: true,
		},
		{
			ip:         net.IPv4bcast,
			expectIPv4: true,
		},
		{
			ip:         ParseIPSloppy("127.0.0.1"),
			expectIPv4: true,
		},
		{
			ip:         ParseIPSloppy("10.20.40.40"),
			expectIPv4: true,
		},
		{
			ip:         ParseIPSloppy("172.17.3.0"),
			expectIPv4: true,
		},
		{
			ip:         nil,
			expectIPv4: false,
		},
		{
			ip:         net.IPv6loopback,
			expectIPv4: false,
		},
		{
			ip:         net.IPv6zero,
			expectIPv4: false,
		},
		{
			ip:         ParseIPSloppy("fd00::600d:f00d"),
			expectIPv4: false,
		},
		{
			ip:         ParseIPSloppy("2001:db8::5"),
			expectIPv4: false,
		},
	}
	for i := range testCases {
		isIPv4 := IsIPv4(testCases[i].ip)
		if isIPv4 != testCases[i].expectIPv4 {
			t.Errorf("[%d] Expect ipv4 %v, got %v", i+1, testCases[i].expectIPv4, isIPv4)
		}
	}
}

func TestIsIPv4CIDRString(t *testing.T) {
	testCases := []struct {
		desc         string
		cidr         string
		expectResult bool
	}{
		{
			desc:         "ipv4 CIDR 1",
			cidr:         "10.0.0.0/8",
			expectResult: true,
		},
		{
			desc:         "ipv4 CIDR 2",
			cidr:         "192.168.0.0/16",
			expectResult: true,
		},
		{
			desc:         "ipv6 CIDR 1",
			cidr:         "::/1",
			expectResult: false,
		},
		{
			desc:         "ipv6 CIDR 2",
			cidr:         "2000::/10",
			expectResult: false,
		},
		{
			desc:         "ipv6 CIDR 3",
			cidr:         "2001:db8::/32",
			expectResult: false,
		},
	}

	for _, tc := range testCases {
		res := IsIPv4CIDRString(tc.cidr)
		if res != tc.expectResult {
			t.Errorf("%v: want IsIPv4CIDRString=%v, got %v", tc.desc, tc.expectResult, res)
		}
	}
}

func TestIsIPv4CIDR(t *testing.T) {
	testCases := []struct {
		desc         string
		cidr         string
		expectResult bool
	}{
		{
			desc:         "ipv4 CIDR 1",
			cidr:         "10.0.0.0/8",
			expectResult: true,
		},
		{
			desc:         "ipv4 CIDR 2",
			cidr:         "192.168.0.0/16",
			expectResult: true,
		},
		{
			desc:         "ipv6 CIDR 1",
			cidr:         "::/1",
			expectResult: false,
		},
		{
			desc:         "ipv6 CIDR 2",
			cidr:         "2000::/10",
			expectResult: false,
		},
		{
			desc:         "ipv6 CIDR 3",
			cidr:         "2001:db8::/32",
			expectResult: false,
		},
	}

	for _, tc := range testCases {
		_, cidr, _ := ParseCIDRSloppy(tc.cidr)
		res := IsIPv4CIDR(cidr)
		if res != tc.expectResult {
			t.Errorf("%v: want IsIPv4CIDR=%v, got %v", tc.desc, tc.expectResult, res)
		}
	}
}

func TestParsePort(t *testing.T) {
	var tests = []struct {
		name          string
		port          string
		allowZero     bool
		expectedPort  int
		expectedError bool
	}{
		{
			name:         "valid port: 1",
			port:         "1",
			expectedPort: 1,
		},
		{
			name:         "valid port: 1234",
			port:         "1234",
			expectedPort: 1234,
		},
		{
			name:         "valid port: 65535",
			port:         "65535",
			expectedPort: 65535,
		},
		{
			name:          "invalid port: not a number",
			port:          "a",
			expectedError: true,
			allowZero:     false,
		},
		{
			name:          "invalid port: too small",
			port:          "0",
			expectedError: true,
		},
		{
			name:          "invalid port: negative",
			port:          "-10",
			expectedError: true,
		},
		{
			name:          "invalid port: too big",
			port:          "65536",
			expectedError: true,
		},
		{
			name:      "zero port: allowed",
			port:      "0",
			allowZero: true,
		},
		{
			name:          "zero port: not allowed",
			port:          "0",
			expectedError: true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			actualPort, actualError := ParsePort(rt.port, rt.allowZero)

			if actualError != nil && !rt.expectedError {
				t.Errorf("%s unexpected failure: %v", rt.name, actualError)
				return
			}
			if actualError == nil && rt.expectedError {
				t.Errorf("%s passed when expected to fail", rt.name)
				return
			}
			if actualPort != rt.expectedPort {
				t.Errorf("%s returned wrong port: got %d, expected %d", rt.name, actualPort, rt.expectedPort)
			}
		})
	}
}

func TestRangeSize(t *testing.T) {
	testCases := []struct {
		name  string
		cidr  string
		addrs int64
	}{
		{
			name:  "supported IPv4 cidr",
			cidr:  "192.168.1.0/24",
			addrs: 256,
		},
		{
			name:  "unsupported IPv4 cidr",
			cidr:  "192.168.1.0/1",
			addrs: 0,
		},
		{
			name:  "unsupported IPv6 mask",
			cidr:  "2001:db8::/1",
			addrs: 0,
		},
	}

	for _, tc := range testCases {
		_, cidr, err := ParseCIDRSloppy(tc.cidr)
		if err != nil {
			t.Errorf("failed to parse cidr for test %s, unexpected error: '%s'", tc.name, err)
		}
		if size := RangeSize(cidr); size != tc.addrs {
			t.Errorf("test %s failed. %s should have a range size of %d, got %d",
				tc.name, tc.cidr, tc.addrs, size)
		}
	}
}

func TestGetIndexedIP(t *testing.T) {
	testCases := []struct {
		cidr        string
		index       int
		expectError bool
		expectedIP  string
	}{
		{
			cidr:        "192.168.1.0/24",
			index:       20,
			expectError: false,
			expectedIP:  "192.168.1.20",
		},
		{
			cidr:        "192.168.1.0/30",
			index:       10,
			expectError: true,
		},
		{
			cidr:        "192.168.1.0/24",
			index:       255,
			expectError: false,
			expectedIP:  "192.168.1.255",
		},
		{
			cidr:        "255.255.255.0/24",
			index:       256,
			expectError: true,
		},
		{
			cidr:        "fd:11:b2:be::/120",
			index:       20,
			expectError: false,
			expectedIP:  "fd:11:b2:be::14",
		},
		{
			cidr:        "fd:11:b2:be::/126",
			index:       10,
			expectError: true,
		},
		{
			cidr:        "fd:11:b2:be::/120",
			index:       255,
			expectError: false,
			expectedIP:  "fd:11:b2:be::ff",
		},
		{
			cidr:        "00:00:00:be::/120",
			index:       255,
			expectError: false,
			expectedIP:  "::be:0:0:0:ff",
		},
	}

	for _, tc := range testCases {
		_, subnet, err := ParseCIDRSloppy(tc.cidr)
		if err != nil {
			t.Errorf("failed to parse cidr %s, unexpected error: '%s'", tc.cidr, err)
		}

		ip, err := GetIndexedIP(subnet, tc.index)
		if err == nil && tc.expectError || err != nil && !tc.expectError {
			t.Errorf("expectedError is %v and err is %s", tc.expectError, err)
			continue
		}

		if err == nil {
			ipString := ip.String()
			if ipString != tc.expectedIP {
				t.Errorf("expected %s but instead got %s", tc.expectedIP, ipString)
			}
		}

	}
}
