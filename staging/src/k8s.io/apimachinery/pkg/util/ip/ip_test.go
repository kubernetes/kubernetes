/*
Copyright 2023 The Kubernetes Authors.

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

func TestParseIP(t *testing.T) {
	for _, tc := range []struct {
		name   string
		input  string
		legacy bool
		output netip.Addr
		clean  netip.Addr
	}{
		{
			name:   "ipv4",
			input:  "1.2.3.4",
			output: netip.MustParseAddr("1.2.3.4"),
		},
		{
			name:   "ipv4 with leading 0s",
			input:  "001.002.003.004",
			legacy: true,
			output: netip.MustParseAddr("1.2.3.4"),
		},
		{
			name:   "ipv4-in-ipv6",
			input:  "::ffff:1.2.3.4",
			legacy: true,
			output: netip.MustParseAddr("::ffff:1.2.3.4"),
			clean:  netip.MustParseAddr("1.2.3.4"),
		},
		{
			name:   "ipv6",
			input:  "1234::abcd",
			output: netip.MustParseAddr("1234::abcd"),
		},
		{
			name:   "ipv6 non-canonical 1",
			input:  "1234:0::abcd",
			output: netip.MustParseAddr("1234::abcd"),
		},
		{
			name:   "ipv6 non-canonical 2",
			input:  "1234::ABCD",
			output: netip.MustParseAddr("1234::abcd"),
		},
		{
			name:  "junk (invalid)",
			input: "blah",
		},
		{
			name:  "cidr (invalid)",
			input: "1.2.3.0/24",
		},
		{
			name:  "IPv4 with out-of-range octets (invalid)",
			input: "1.2.3.400",
		},
		{
			name:  "IPv6 with out-of-range segment (invalid)",
			input: "2001:db8::10005",
		},
		{
			name:  "IPv4:port (invalid)",
			input: "1.2.3.4:80",
		},
		{
			name:  "[IPv6] with brackets (invalid)",
			input: "[2001:db8::5]",
		},
		{
			name:  "[IPv6]:port (invalid)",
			input: "[2001:db8::5]:80",
		},
		{
			name:   "ipv6 with zone (invalid)",
			input:  "1234::abcd%eth0",
		},
		{
			name:  "leading whitespace (invalid)",
			input: " 1.2.3.4",
		},
		{
			name:  "trailing whitespace (invalid)",
			input: "1.2.3.4 ",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !tc.clean.IsValid() {
				tc.clean = tc.output
			}

			ip, err := ParseIP(tc.input)
			if !tc.output.IsValid() || tc.legacy {
				if err == nil {
					t.Errorf("expected %q to not parse with ParseIP but got %q", tc.input, ip.String())
				}
			} else if err != nil {
				t.Errorf("expected %q to parse with ParseIP but got %v", tc.input, err)
			} else if ip != tc.output {
				t.Errorf("expected %q to parse to %q with ParseIP but got %q", tc.input, tc.output.String(), ip.String())
			}

			ip, cleanIP, err := ParseLegacyIP(tc.input, "unit test")
			if !tc.output.IsValid() {
				if err == nil {
					t.Errorf("expected %q to not parse with ParseLegacyIP but got %q", tc.input, ip.String())
				}
			} else if err != nil {
				t.Errorf("expected %q to parse with ParseLegacyIP but got %v", tc.input, err)
			} else {
				if ip != tc.output {
					t.Errorf("expected %q to parse to %q with ParseLegacyIP but got %q", tc.input, tc.output.String(), ip.String())
				}
				if cleanIP != tc.clean {
					t.Errorf("expected %q to parse to clean value %q with ParseLegacyIP but got %q", tc.input, tc.clean.String(), cleanIP.String())
				}
			}

			ip, err = ParseCanonicalIP(tc.input)
			if tc.output.String() != tc.input || tc.legacy {
				if err == nil {
					t.Errorf("expected %q to not parse with ParseCanonicalIP but got %q", tc.input, ip.String())
				}
			} else if err != nil {
				t.Errorf("expected %q to parse with ParseCanonicalIP but got %v", tc.input, err)
			} else if ip != tc.output {
				t.Errorf("expected %q to parse to %q with ParseCanonicalIP but got %q", tc.input, tc.output.String(), ip.String())
			}
		})
	}
}

func TestParseIPList(t *testing.T) {
	for _, tc := range []struct {
		name   string
		input  string
		legacy bool
		output []netip.Addr
	}{
		{
			name:   "empty",
			input:  "",
			output: []netip.Addr{},
		},
		{
			name:  "ipv4 single",
			input: "1.2.3.4",
			output: []netip.Addr{
				netip.MustParseAddr("1.2.3.4"),
			},
		},
		{
			name:  "ipv4 double",
			input: "1.2.3.4,5.6.7.8",
			output: []netip.Addr{
				netip.MustParseAddr("1.2.3.4"),
				netip.MustParseAddr("5.6.7.8"),
			},
		},
		{
			name:  "ipv4 triple",
			input: "1.2.3.4,5.6.7.8,9.10.11.12",
			output: []netip.Addr{
				netip.MustParseAddr("1.2.3.4"),
				netip.MustParseAddr("5.6.7.8"),
				netip.MustParseAddr("9.10.11.12"),
			},
		},
		{
			name:   "ipv4 legacy",
			input:  "001.002.003.004",
			legacy: true,
			output: []netip.Addr{
				netip.MustParseAddr("1.2.3.4"),
			},
		},
		{
			name:   "ipv4 mixed legacy",
			input:  "1.2.3.4,001.002.003.004,5.6.7.8",
			legacy: true,
			output: []netip.Addr{
				netip.MustParseAddr("1.2.3.4"),
				netip.MustParseAddr("1.2.3.4"),
				netip.MustParseAddr("5.6.7.8"),
			},
		},
		{
			name:  "mixed IP family",
			input: "1234::abcd,5.6.7.8",
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
			ips, err := ParseIPList(tc.input)
			if tc.output == nil || tc.legacy {
				if err == nil {
					t.Errorf("expected %q to not parse with ParseIPList but got %v", tc.input, ips)
				}
			} else if err != nil {
				t.Errorf("expected %q to parse with ParseIPList but got %v", tc.input, err)
			} else if !reflect.DeepEqual(ips, tc.output) {
				t.Errorf("expected %q to parse to %v with ParseIPList but got %v", tc.input, tc.output, ips)
			}

			ips, _, err = ParseLegacyIPList(tc.input, "unit test")
			if tc.output == nil {
				if err == nil {
					t.Errorf("expected %q to not parse with ParseLegacyIPList but got %q", tc.input, ips)
				}
			} else if err != nil {
				t.Errorf("expected %q to parse with ParseLegacyIPList but got %v", tc.input, err)
			} else if !reflect.DeepEqual(ips, tc.output) {
				t.Errorf("expected %q to parse to %v with ParseLegacyIPList but got %v", tc.input, tc.output, ips)
			}
		})
	}
}
