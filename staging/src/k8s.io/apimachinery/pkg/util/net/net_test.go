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
	"reflect"
	"testing"
)

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
			ip:         net.ParseIP("127.0.0.1"),
			expectIPv6: false,
		},
		{
			ip:         net.ParseIP("10.20.40.40"),
			expectIPv6: false,
		},
		{
			ip:         net.ParseIP("172.17.3.0"),
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
			ip:         net.ParseIP("fd00::600d:f00d"),
			expectIPv6: true,
		},
		{
			ip:         net.ParseIP("2001:db8::5"),
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
		res := IsIPv6CIDR(tc.cidr)
		if res != tc.expectResult {
			t.Errorf("%v: want IsIPv6CIDR=%v, got %v", tc.desc, tc.expectResult, res)
		}
	}
}

func TestFilterIncorrectIPVersion(t *testing.T) {
	testCases := []struct {
		desc             string
		isIPv6           bool
		ipStrings        []string
		expectCorrects   []string
		expectIncorrects []string
	}{
		{
			desc:             "all ipv4 strings in ipv4 mode",
			isIPv6:           false,
			ipStrings:        []string{"10.0.0.1", "192.168.0.1", "127.0.0.1"},
			expectCorrects:   []string{"10.0.0.1", "192.168.0.1", "127.0.0.1"},
			expectIncorrects: nil,
		},
		{
			desc:             "all ipv6 strings in ipv4 mode",
			isIPv6:           false,
			ipStrings:        []string{"::1", "fd00::600d:f00d", "2001:db8::5"},
			expectCorrects:   nil,
			expectIncorrects: []string{"::1", "fd00::600d:f00d", "2001:db8::5"},
		},
		{
			desc:             "mixed versions in ipv4 mode",
			isIPv6:           false,
			ipStrings:        []string{"10.0.0.1", "192.168.0.1", "127.0.0.1", "::1", "fd00::600d:f00d", "2001:db8::5"},
			expectCorrects:   []string{"10.0.0.1", "192.168.0.1", "127.0.0.1"},
			expectIncorrects: []string{"::1", "fd00::600d:f00d", "2001:db8::5"},
		},
		{
			desc:             "all ipv4 strings in ipv6 mode",
			isIPv6:           true,
			ipStrings:        []string{"10.0.0.1", "192.168.0.1", "127.0.0.1"},
			expectCorrects:   nil,
			expectIncorrects: []string{"10.0.0.1", "192.168.0.1", "127.0.0.1"},
		},
		{
			desc:             "all ipv6 strings in ipv6 mode",
			isIPv6:           true,
			ipStrings:        []string{"::1", "fd00::600d:f00d", "2001:db8::5"},
			expectCorrects:   []string{"::1", "fd00::600d:f00d", "2001:db8::5"},
			expectIncorrects: nil,
		},
		{
			desc:             "mixed versions in ipv6 mode",
			isIPv6:           true,
			ipStrings:        []string{"10.0.0.1", "192.168.0.1", "127.0.0.1", "::1", "fd00::600d:f00d", "2001:db8::5"},
			expectCorrects:   []string{"::1", "fd00::600d:f00d", "2001:db8::5"},
			expectIncorrects: []string{"10.0.0.1", "192.168.0.1", "127.0.0.1"},
		},
	}

	for _, tc := range testCases {
		corrects, incorrects := FilterIncorrectIPVersion(tc.ipStrings, tc.isIPv6)
		if !reflect.DeepEqual(tc.expectCorrects, corrects) {
			t.Errorf("%v: want corrects=%v, got %v", tc.desc, tc.expectCorrects, corrects)
		}
		if !reflect.DeepEqual(tc.expectIncorrects, incorrects) {
			t.Errorf("%v: want incorrects=%v, got %v", tc.desc, tc.expectIncorrects, incorrects)
		}
	}
}

func TestFilterIncorrectCIDRVersion(t *testing.T) {
	testCases := []struct {
		desc             string
		isIPv6           bool
		cidrStrings      []string
		expectCorrects   []string
		expectIncorrects []string
	}{
		{
			desc:             "all ipv4 strings in ipv4 mode",
			isIPv6:           false,
			cidrStrings:      []string{"0.0.0.0/1", "1.0.0.0/1"},
			expectCorrects:   []string{"0.0.0.0/1", "1.0.0.0/1"},
			expectIncorrects: nil,
		},
		{
			desc:             "all ipv6 strings in ipv4 mode",
			isIPv6:           false,
			cidrStrings:      []string{"2001:db8::/32", "2001:0db8:0123:4567::/64"},
			expectCorrects:   nil,
			expectIncorrects: []string{"2001:db8::/32", "2001:0db8:0123:4567::/64"},
		},
		{
			desc:             "mixed versions in ipv4 mode",
			isIPv6:           false,
			cidrStrings:      []string{"0.0.0.0/1", "1.0.0.0/1", "2001:db8::/32", "2001:0db8:0123:4567::/64"},
			expectCorrects:   []string{"0.0.0.0/1", "1.0.0.0/1"},
			expectIncorrects: []string{"2001:db8::/32", "2001:0db8:0123:4567::/64"},
		},
		{
			desc:             "all ipv4 strings in ipv6 mode",
			isIPv6:           true,
			cidrStrings:      []string{"0.0.0.0/1", "1.0.0.0/1"},
			expectCorrects:   nil,
			expectIncorrects: []string{"0.0.0.0/1", "1.0.0.0/1"},
		},
		{
			desc:             "all ipv6 strings in ipv6 mode",
			isIPv6:           true,
			cidrStrings:      []string{"2001:db8::/32", "2001:0db8:0123:4567::/64"},
			expectCorrects:   []string{"2001:db8::/32", "2001:0db8:0123:4567::/64"},
			expectIncorrects: nil,
		},
		{
			desc:             "mixed versions in ipv6 mode",
			isIPv6:           true,
			cidrStrings:      []string{"0.0.0.0/1", "1.0.0.0/1", "2001:db8::/32", "2001:0db8:0123:4567::/64"},
			expectCorrects:   []string{"2001:db8::/32", "2001:0db8:0123:4567::/64"},
			expectIncorrects: []string{"0.0.0.0/1", "1.0.0.0/1"},
		},
	}

	for _, tc := range testCases {
		corrects, incorrects := FilterIncorrectCIDRVersion(tc.cidrStrings, tc.isIPv6)
		if !reflect.DeepEqual(tc.expectCorrects, corrects) {
			t.Errorf("%v: want corrects=%v, got %v", tc.desc, tc.expectCorrects, corrects)
		}
		if !reflect.DeepEqual(tc.expectIncorrects, incorrects) {
			t.Errorf("%v: want incorrects=%v, got %v", tc.desc, tc.expectIncorrects, incorrects)
		}
	}
}
