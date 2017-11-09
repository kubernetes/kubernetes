/*
Copyright 2017 The Kubernetes Authors.

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
)

func TestIPPart(t *testing.T) {
	const noError = ""

	testCases := []struct {
		endpoint      string
		expectedIP    string
		expectedError string
	}{
		{"1.2.3.4", "1.2.3.4", noError},
		{"1.2.3.4:9999", "1.2.3.4", noError},
		{"2001:db8::1:1", "2001:db8::1:1", noError},
		{"[2001:db8::2:2]:9999", "2001:db8::2:2", noError},
		{"1.2.3.4::9999", "", "too many colons"},
		{"1.2.3.4:[0]", "", "unexpected '[' in address"},
	}

	for _, tc := range testCases {
		ip := IPPart(tc.endpoint)
		if tc.expectedError == noError {
			if ip != tc.expectedIP {
				t.Errorf("Unexpected IP for %s: Expected: %s, Got %s", tc.endpoint, tc.expectedIP, ip)
			}
		} else if ip != "" {
			t.Errorf("Error did not occur for %s, expected: '%s' error", tc.endpoint, tc.expectedError)
		}
	}
}

func TestToCIDR(t *testing.T) {
	testCases := []struct {
		ip           string
		expectedAddr string
	}{
		{"1.2.3.4", "1.2.3.4/32"},
		{"2001:db8::1:1", "2001:db8::1:1/128"},
	}

	for _, tc := range testCases {
		ip := net.ParseIP(tc.ip)
		addr := ToCIDR(ip)
		if addr != tc.expectedAddr {
			t.Errorf("Unexpected host address for %s: Expected: %s, Got %s", tc.ip, tc.expectedAddr, addr)
		}
	}
}
