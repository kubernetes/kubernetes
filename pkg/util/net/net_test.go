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
