/*
Copyright 2021 The Kubernetes Authors.

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

func TestIPToDecimal(t *testing.T) {
	tests := []struct {
		name string
		ip   net.IP
		want string
	}{
		{
			name: "empty",
			ip:   nil,
			want: "",
		},
		{
			name: "ipv4",
			ip:   net.ParseIP("192.168.1.1"),
			want: "3232235777",
		},
		{
			name: "ipv6",
			ip:   net.ParseIP("2001:db2::2"),
			want: "42540765935913617771317959390390124546",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IPToDecimal(tt.ip); got != tt.want {
				t.Errorf("IPToDecimal() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDecimalToIP(t *testing.T) {
	tests := []struct {
		name      string
		ipDecimal string
		want      net.IP
	}{
		{
			name:      "empty",
			ipDecimal: "",
			want:      nil,
		},
		{
			name:      "not a number",
			ipDecimal: "asdasddsa",
			want:      nil,
		},
		{
			name:      "more than 16 bytes number",
			ipDecimal: "4254076593591361777131795939039012454642540765935913617771317959390390124546",
			want:      nil,
		},
		{
			name:      "ipv4",
			ipDecimal: "3232235777",
			want:      net.ParseIP("192.168.1.1"),
		},
		{
			name:      "ipv6",
			ipDecimal: "42540765935913617771317959390390124546",
			want:      net.ParseIP("2001:db2::2"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DecimalToIP(tt.ipDecimal)
			if !tt.want.Equal(got) {
				t.Errorf("DecimalToIP() = %v, want %v", got, tt.want)
			}
		})
	}
}
