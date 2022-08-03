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

package disablecompression

import (
	"net/http"
	"testing"
)

func TestNewClientIPPredicate(t *testing.T) {
	tests := []struct {
		name        string
		cidrStrings []string
		wantErr     bool
	}{
		{
			name:        "rfc1918",
			cidrStrings: []string{"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"},
		},
		{
			name:        "rfc4193 (ipv6)",
			cidrStrings: []string{"fc00::/7"},
		},
		{
			name:        "ipv6 loopback",
			cidrStrings: []string{"::1/128"},
		},
		{
			name:        "bad cidr",
			cidrStrings: []string{"not a cidr string"},
			wantErr:     true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewClientIPPredicate(tt.cidrStrings)
			if (err != nil) != tt.wantErr {
				t.Fatalf("NewClientIPPredicate() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if got, want := len(got.cidrs), len(tt.cidrStrings); got != want {
				t.Errorf("len(NewClientIPPredicate.cidrs()) = %v, want %v", got, want)
			}
		})
	}
}

func TestClientIPPredicate_Predicate(t *testing.T) {
	check, err := NewClientIPPredicate([]string{"::1/128", "10.0.0.0/8"})
	if err != nil {
		t.Fatalf("failed to construct NewClientIPPredicate: %v", err)
	}
	tests := []struct {
		name    string
		req     *http.Request
		want    bool
		wantErr bool
	}{
		{
			name: "ipv4, in range",
			req:  &http.Request{RemoteAddr: "10.0.0.1:123"},
			want: true,
		},
		{
			name: "ipv4, out of range",
			req:  &http.Request{RemoteAddr: "11.0.0.1:123"},
			want: false,
		},
		{
			name: "ipv6, in range",
			req:  &http.Request{RemoteAddr: "[::1]:123"},
			want: true,
		},
		{
			name: "ipv6, out of range",
			req:  &http.Request{RemoteAddr: "[::2]:123"},
			want: false,
		},
		{
			name:    "no IP",
			req:     &http.Request{},
			wantErr: true,
		},
		{
			name:    "RemoteAddr doesn't parse",
			req:     &http.Request{RemoteAddr: "this is definitely not an IP address and port"},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := check.Predicate(tt.req)
			if (err != nil) != tt.wantErr {
				t.Errorf("ClientIPPredicate.Predicate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("ClientIPPredicate.Predicate() = %v, want %v", got, tt.want)
			}
		})
	}
}
