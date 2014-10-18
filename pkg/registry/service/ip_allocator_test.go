/*
Copyright 2014 Google Inc. All rights reserved.

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

package service

import (
	"net"
	"testing"
)

func TestNew(t *testing.T) {
	if newIPAllocator(nil) != nil {
		t.Errorf("expected nil")
	}
	if newIPAllocator(&net.IPNet{}) != nil {
		t.Errorf("expected nil")
	}
	_, ipnet, err := net.ParseCIDR("93.76.0.0/22")
	if err != nil {
		t.Error(err)
	}
	ipa := newIPAllocator(ipnet)
	if ipa == nil {
		t.Errorf("expected non-nil")
	}
	if len(ipa.used) != 128 { // a /22 has 1024 IPs, 8 per byte = 128
		t.Errorf("wrong size for ipa.used")
	}
	if ipa.used[0] != 0x01 {
		t.Errorf("network address was not reserved")
	}
	if ipa.used[127] != 0x80 {
		t.Errorf("broadcast address was not reserved")
	}
}

func TestAllocate(t *testing.T) {
	_, ipnet, _ := net.ParseCIDR("93.76.0.0/22")
	ipa := newIPAllocator(ipnet)

	if err := ipa.Allocate(net.ParseIP("93.76.0.1")); err != nil {
		t.Errorf("expected success, got %s", err)
	}

	if ipa.Allocate(net.ParseIP("93.76.0.1")) == nil {
		t.Errorf("expected failure")
	}

	if ipa.Allocate(net.ParseIP("1.2.3.4")) == nil {
		t.Errorf("expected failure")
	}
}

func TestAllocateNext(t *testing.T) {
	_, ipnet, _ := net.ParseCIDR("93.76.0.0/22")
	ipa := newIPAllocator(ipnet)

	ip1, err := ipa.AllocateNext()
	if err != nil {
		t.Error(err)
	}
	if !ip1.Equal(net.ParseIP("93.76.0.1")) {
		t.Errorf("expected 93.76.0.1, got %s", ip1)
	}

	ip2, err := ipa.AllocateNext()
	if err != nil {
		t.Error(err)
	}
	if !ip2.Equal(net.ParseIP("93.76.0.2")) {
		t.Errorf("expected 93.76.0.2, got %s", ip2)
	}

	// Burn a bunch of addresses.
	for i := 3; i < 256; i++ {
		ipa.AllocateNext()
	}

	ip256, err := ipa.AllocateNext()
	if err != nil {
		t.Error(err)
	}
	if !ip256.Equal(net.ParseIP("93.76.1.0")) {
		t.Errorf("expected 93.76.1.0, got %s", ip256)
	}

	ip257, err := ipa.AllocateNext()
	if err != nil {
		t.Error(err)
	}
	if !ip257.Equal(net.ParseIP("93.76.1.1")) {
		t.Errorf("expected 93.76.1.1, got %s", ip257)
	}

	// Burn a bunch of addresses.
	for i := 258; i < 1022; i++ {
		ipa.AllocateNext()
	}

	ip1022, err := ipa.AllocateNext()
	if err != nil {
		t.Error(err)
	}
	if !ip1022.Equal(net.ParseIP("93.76.3.254")) {
		t.Errorf("expected 93.76.3.254, got %s", ip1022)
	}

	_, err = ipa.AllocateNext()
	if err == nil {
		t.Errorf("Expected nil - allocator is full")
	}
}

func TestRelease(t *testing.T) {
	_, ipnet, _ := net.ParseCIDR("93.76.0.0/24")
	ipa := newIPAllocator(ipnet)

	err := ipa.Release(net.ParseIP("1.2.3.4"))
	if err == nil {
		t.Errorf("Expected an error")
	}

	ip1, err := ipa.AllocateNext()
	if err != nil {
		t.Error(err)
	}
	ip2, err := ipa.AllocateNext()
	if err != nil {
		t.Error(err)
	}
	_, err = ipa.AllocateNext()
	if err != nil {
		t.Error(err)
	}

	err = ipa.Release(ip2)
	if err != nil {
		t.Error(err)
	}

	ip4, err := ipa.AllocateNext()
	if !ip4.Equal(ip2) {
		t.Errorf("Expected %s, got %s", ip2, ip4)
	}

	// Burn a bunch of addresses.
	for i := 4; i < 255; i++ {
		ipa.AllocateNext()
	}
	_, err = ipa.AllocateNext()
	if err == nil {
		t.Errorf("Expected an error")
	}
	ipa.Release(ip1)

	ip5, err := ipa.AllocateNext()
	if !ip5.Equal(ip1) {
		t.Errorf("Expected %s, got %s", ip1, ip5)
	}
}

func TestFFS(t *testing.T) {
	_, err := ffs(0)
	if err == nil {
		t.Errorf("Expected error")
	}

	testCases := []struct {
		value    byte
		expected uint
	}{
		{0x01, 0}, {0x02, 1}, {0x04, 2}, {0x08, 3},
		{0x10, 4}, {0x20, 5}, {0x40, 6}, {0x80, 7},
		{0x22, 1}, {0xa0, 5}, {0xfe, 1}, {0xff, 0},
	}
	for _, tc := range testCases {
		r, err := ffs(tc.value)
		if err != nil {
			t.Error(err)
		}
		if r != tc.expected {
			t.Errorf("Expected %d, got %d", tc.expected, r)
		}
	}
}

func TestIPAdd(t *testing.T) {
	testCases := []struct {
		ip       string
		offset   int
		expected string
	}{
		{"1.2.3.0", 0, "1.2.3.0"},
		{"1.2.3.0", 1, "1.2.3.1"},
		{"1.2.3.0", 255, "1.2.3.255"},
		{"1.2.3.1", 255, "1.2.4.0"},
		{"1.2.3.2", 255, "1.2.4.1"},
		{"1.2.3.0", 256, "1.2.4.0"},
		{"1.2.3.0", 257, "1.2.4.1"},
		{"1.2.3.0", 65536, "1.3.3.0"},
		{"1.2.3.4", 1, "1.2.3.5"},
		{"255.255.255.255", 1, "0.0.0.0"},
		{"255.255.255.255", 2, "0.0.0.1"},
	}
	for _, tc := range testCases {
		r := ipAdd(net.ParseIP(tc.ip), tc.offset)
		if !r.Equal(net.ParseIP(tc.expected)) {
			t.Errorf("Expected %s, got %s", tc.expected, r)
		}
	}
}

func TestIPSub(t *testing.T) {
	testCases := []struct {
		lhs      string
		rhs      string
		expected int
	}{
		{"1.2.3.0", "1.2.3.0", 0},
		{"1.2.3.1", "1.2.3.0", 1},
		{"1.2.3.255", "1.2.3.0", 255},
		{"1.2.4.0", "1.2.3.0", 256},
		{"1.2.4.0", "1.2.3.255", 1},
		{"1.2.4.1", "1.2.3.0", 257},
		{"1.3.3.0", "1.2.3.0", 65536},
		{"1.2.3.5", "1.2.3.4", 1},
		{"0.0.0.0", "0.0.0.1", -1},
		{"0.0.1.0", "0.0.0.1", 255},
	}
	for _, tc := range testCases {
		r := ipSub(net.ParseIP(tc.lhs), net.ParseIP(tc.rhs))
		if r != tc.expected {
			t.Errorf("Expected %v, got %v", tc.expected, r)
		}
	}
}

func TestCopyIP(t *testing.T) {
	ip1 := net.ParseIP("1.2.3.4")
	ip2 := copyIP(ip1)
	ip2[0]++
	if ip1[0] == ip2[0] {
		t.Errorf("copyIP did not copy")
	}
}

func TestSimplifyIP(t *testing.T) {
	ip4 := net.ParseIP("1.2.3.4")
	if len(ip4) != 16 {
		t.Errorf("expected 16 bytes")
	}
	if len(simplifyIP(ip4)) != 4 {
		t.Errorf("expected 4 bytes")
	}
	ip6 := net.ParseIP("::1.2.3.4")
	if len(ip6) != 16 {
		t.Errorf("expected 16 bytes")
	}
	if len(simplifyIP(ip6)) != 16 {
		t.Errorf("expected 16 bytes")
	}
	if simplifyIP([]byte{0, 0}) != nil {
		t.Errorf("expected nil")
	}
}
