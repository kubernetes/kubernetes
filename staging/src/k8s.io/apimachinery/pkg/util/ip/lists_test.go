/*
Copyright 2024 The Kubernetes Authors.

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
	"reflect"
	"testing"
)

func TestParseIPList(t *testing.T) {
	for _, tc := range []struct {
		name   string
		input  string
		valid  bool
		output []netip.Addr
	}{
		{
			name:   "empty",
			input:  "",
			valid:  true,
			output: []netip.Addr{},
		},
		{
			name:  "ipv4 single",
			input: "1.2.3.4",
			valid: true,
			output: []netip.Addr{
				netip.MustParseAddr("1.2.3.4"),
			},
		},
		{
			name:  "ipv4 double",
			input: "1.2.3.4,5.6.7.8",
			valid: true,
			output: []netip.Addr{
				netip.MustParseAddr("1.2.3.4"),
				netip.MustParseAddr("5.6.7.8"),
			},
		},
		{
			name:  "ipv4 triple",
			input: "1.2.3.4,5.6.7.8,9.10.11.12",
			valid: true,
			output: []netip.Addr{
				netip.MustParseAddr("1.2.3.4"),
				netip.MustParseAddr("5.6.7.8"),
				netip.MustParseAddr("9.10.11.12"),
			},
		},
		{
			name:  "ipv4 legacy",
			input: "001.002.003.004",
			valid: false,
			output: []netip.Addr{
				netip.MustParseAddr("1.2.3.4"),
			},
		},
		{
			name:  "ipv4 mixed legacy",
			input: "1.2.3.4,001.002.003.004,5.6.7.8",
			valid: false,
			output: []netip.Addr{
				netip.MustParseAddr("1.2.3.4"),
				netip.MustParseAddr("1.2.3.4"),
				netip.MustParseAddr("5.6.7.8"),
			},
		},
		{
			name:  "mixed IP family",
			input: "1234::abcd,5.6.7.8",
			valid: true,
			output: []netip.Addr{
				netip.MustParseAddr("1234::abcd"),
				netip.MustParseAddr("5.6.7.8"),
			},
		},
		{
			name:   "invalid",
			input:  "blah",
			output: nil,
		},
		{
			name:   "mixed invalid",
			input:  "1.2.3.4,blah",
			output: nil,
		},
		{
			name:   "extra commas",
			input:  "1.2.3.4,,5.6.7.8",
			output: nil,
		},
		{
			name:   "whitespace",
			input:  "1.2.3.4, 5.6.7.8",
			output: nil,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ips, err := Split(tc.input, ",", ParseValidIP)
			if tc.output == nil || !tc.valid {
				if err == nil {
					t.Errorf("expected %q to not parse with ParseValidIP but got %v", tc.input, ips)
				}
			} else if err != nil {
				t.Errorf("expected %q to parse with ParseValidIP but got %v", tc.input, err)
			} else if !reflect.DeepEqual(ips, tc.output) {
				t.Errorf("expected %q to parse to %v with ParseValidIP but got %v", tc.input, tc.output, ips)
			}

			ips, err = Split(tc.input, ",", ParseIP)
			if tc.output == nil {
				if err == nil {
					t.Errorf("expected %q to not parse with ParseIP but got %q", tc.input, ips)
				}
			} else if err != nil {
				t.Errorf("expected %q to parse with ParseIP but got %v", tc.input, err)
			} else if !reflect.DeepEqual(ips, tc.output) {
				t.Errorf("expected %q to parse to %v with ParseIP but got %v", tc.input, tc.output, ips)
			}
		})
	}
}

func TestParseCIDRList(t *testing.T) {
	for _, tc := range []struct {
		name   string
		input  string
		valid  bool
		output []netip.Prefix
	}{
		{
			name:   "empty",
			input:  "",
			valid:  true,
			output: []netip.Prefix{},
		},
		{
			name:  "ipv4 single",
			input: "1.2.3.0/24",
			valid: true,
			output: []netip.Prefix{
				netip.MustParsePrefix("1.2.3.0/24"),
			},
		},
		{
			name:  "ipv4 double",
			input: "1.2.3.0/24,5.6.0.0/16",
			valid: true,
			output: []netip.Prefix{
				netip.MustParsePrefix("1.2.3.0/24"),
				netip.MustParsePrefix("5.6.0.0/16"),
			},
		},
		{
			name:  "ipv4 triple",
			input: "1.2.3.0/24,5.6.0.0/16,9.10.11.128/25",
			valid: true,
			output: []netip.Prefix{
				netip.MustParsePrefix("1.2.3.0/24"),
				netip.MustParsePrefix("5.6.0.0/16"),
				netip.MustParsePrefix("9.10.11.128/25"),
			},
		},
		{
			name:  "ipv4 legacy",
			input: "001.002.003.000/24",
			valid: false,
			output: []netip.Prefix{
				netip.MustParsePrefix("1.2.3.0/24"),
			},
		},
		{
			name:  "ipv4 mixed legacy",
			input: "1.2.3.0/24,001.002.003.000/24,5.6.0.0/16",
			valid: false,
			output: []netip.Prefix{
				netip.MustParsePrefix("1.2.3.0/24"),
				netip.MustParsePrefix("1.2.3.0/24"),
				netip.MustParsePrefix("5.6.0.0/16"),
			},
		},
		{
			name:  "mixed IP family",
			input: "1234:abcd::/64,5.6.0.0/16",
			valid: true,
			output: []netip.Prefix{
				netip.MustParsePrefix("1234:abcd::/64"),
				netip.MustParsePrefix("5.6.0.0/16"),
			},
		},
		{
			name:  "bad prefix",
			input: "1.2.3.0/24,5.6.7.0/16",
			valid: false,
			output: []netip.Prefix{
				netip.MustParsePrefix("1.2.3.0/24"),
				netip.MustParsePrefix("5.6.0.0/16"),
			},
		},
		{
			name:   "invalid",
			input:  "blah",
			output: nil,
		},
		{
			name:   "mixed invalid",
			input:  "1.2.3.4,blah",
			output: nil,
		},
		{
			name:   "extra commas",
			input:  "1.2.3.4,,5.6.7.8",
			output: nil,
		},
		{
			name:   "whitespace",
			input:  "1.2.3.4, 5.6.7.8",
			output: nil,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cidrs, err := Split(tc.input, ",", ParseValidCIDR)
			if tc.output == nil || !tc.valid {
				if err == nil {
					t.Errorf("expected %q to not parse with ParseValidCIDR but got %v", tc.input, cidrs)
				}
			} else if err != nil {
				t.Errorf("expected %q to parse with ParseValidCIDR but got %v", tc.input, err)
			} else if !reflect.DeepEqual(cidrs, tc.output) {
				t.Errorf("expected %q to parse to %v with ParseValidCIDR but got %v", tc.input, tc.output, cidrs)
			}

			cidrs, err = Split(tc.input, ",", ParseCIDR)
			if tc.output == nil {
				if err == nil {
					t.Errorf("expected %q to not parse with ParseCIDR but got %q", tc.input, cidrs)
				}
			} else if err != nil {
				t.Errorf("expected %q to parse with ParseCIDR but got %v", tc.input, err)
			} else if !reflect.DeepEqual(cidrs, tc.output) {
				t.Errorf("expected %q to parse to %v with ParseCIDR but got %v", tc.input, tc.output, cidrs)
			}
		})
	}
}
