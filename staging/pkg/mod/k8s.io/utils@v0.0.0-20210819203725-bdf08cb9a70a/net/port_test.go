/*
Copyright 2020 The Kubernetes Authors.

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
	"testing"
)

func ExampleLocalPort() {
	lp, err := NewLocalPort(
		"TCP port",
		"",
		IPv4,
		443,
		TCP,
	)
	if err != nil {
		panic(err)
	}
	port, err := ListenPortOpener.OpenLocalPort(lp)
	if err != nil {
		panic(err)
	}
	port.Close()
}

func TestLocalPortString(t *testing.T) {
	testCases := []struct {
		description string
		ip          string
		family      IPFamily
		port        int
		protocol    Protocol
		expectedStr string
		expectedErr bool
	}{
		{"IPv4 UDP", "1.2.3.4", "", 9999, UDP, `"IPv4 UDP" (1.2.3.4:9999/udp)`, false},
		{"IPv4 TCP", "5.6.7.8", "", 1053, TCP, `"IPv4 TCP" (5.6.7.8:1053/tcp)`, false},
		{"IPv6 TCP", "2001:db8::1", "", 80, TCP, `"IPv6 TCP" ([2001:db8::1]:80/tcp)`, false},
		{"IPv4 TCP, all addresses", "", IPv4, 1053, TCP, `"IPv4 TCP, all addresses" (:1053/tcp4)`, false},
		{"IPv6 TCP, all addresses", "", IPv6, 80, TCP, `"IPv6 TCP, all addresses" (:80/tcp6)`, false},
		{"No ip family TCP, all addresses", "", "", 80, TCP, `"No ip family TCP, all addresses" (:80/tcp)`, false},
		{"IP family mismatch", "2001:db8::2", IPv4, 80, TCP, "", true},
		{"IP family mismatch", "1.2.3.4", IPv6, 80, TCP, "", true},
		{"Unsupported protocol", "2001:db8::2", "", 80, "http", "", true},
		{"Invalid IP", "300", "", 80, TCP, "", true},
		{"Invalid ip family", "", "5", 80, TCP, "", true},
	}

	for _, tc := range testCases {
		lp, err := NewLocalPort(
			tc.description,
			tc.ip,
			tc.family,
			tc.port,
			tc.protocol,
		)
		if tc.expectedErr {
			if err == nil {
				t.Errorf("Expected err when creating LocalPort %v", tc)
			}
			continue
		}
		if err != nil {
			t.Errorf("Unexpected err when creating LocalPort %s", err)
			continue
		}
		str := lp.String()
		if str != tc.expectedStr {
			t.Errorf("Unexpected output for %s, expected: %s, got: %s", tc.description, tc.expectedStr, str)
		}
	}
}
