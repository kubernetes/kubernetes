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
		{"1.2.3:8080", "", "invalid ip part"},
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

func TestPortPart(t *testing.T) {
	tests := []struct {
		name     string
		endpoint string
		want     int
		wantErr  bool
	}{
		{
			"no error parsing from ipv4-ip:port",
			"1.2.3.4:1024",
			1024,
			false,
		},
		{
			"no error parsing from ipv6-ip:port",
			"[2001:db8::2:2]:9999",
			9999,
			false,
		},
		{
			"error: missing port",
			"1.2.3.4",
			-1,
			true,
		},
		{
			"error: invalid port '1-2'",
			"1.2.3.4:1-2",
			-1,
			true,
		},
		{
			"error: invalid port 'port'",
			"100.200.3.4:port",
			-1,
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := PortPart(tt.endpoint)
			if (err != nil) != tt.wantErr {
				t.Errorf("PortPart() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("PortPart() = %v, want %v", got, tt.want)
			}
		})
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
