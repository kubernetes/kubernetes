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
	"strings"
	"testing"
)

func TestParseIP(t *testing.T) {
	for _, tc := range []struct {
		name   string
		input  string
		valid  bool
		output netip.Addr
		err    string
	}{
		{
			name:   "ipv4",
			input:  "1.2.3.4",
			valid:  true,
			output: netip.MustParseAddr("1.2.3.4"),
		},
		{
			name:   "ipv4 with leading 0s",
			input:  "001.002.003.004",
			valid:  false,
			output: netip.MustParseAddr("1.2.3.4"),
		},
		{
			name:   "ipv4-in-ipv6",
			input:  "::ffff:1.2.3.4",
			valid:  false,
			output: netip.MustParseAddr("1.2.3.4"),
		},
		{
			name:   "ipv6",
			input:  "1234::abcd",
			valid:  true,
			output: netip.MustParseAddr("1234::abcd"),
		},
		{
			name:   "ipv6 non-canonical 1",
			input:  "1234:0::abcd",
			valid:  true,
			output: netip.MustParseAddr("1234::abcd"),
		},
		{
			name:   "ipv6 non-canonical 2",
			input:  "1234::ABCD",
			valid:  true,
			output: netip.MustParseAddr("1234::abcd"),
		},
		{
			name:  "empty string (invalid)",
			input: "",
			err:   "should not be empty",
		},
		{
			name:  "junk (invalid)",
			input: "blah",
			err:   "unable to parse IP",
		},
		{
			name:  "cidr (invalid)",
			input: "1.2.3.0/24",
			err:   "but got CIDR value",
		},
		{
			name:  "IPv4 with out-of-range octets (invalid)",
			input: "1.2.3.400",
			err:   "IPv4 field has value >255",
		},
		{
			name:  "IPv6 with out-of-range segment (invalid)",
			input: "2001:db8::10005",
			err:   "IPv6 field has value >=2^16",
		},
		{
			name:  "IPv4:port (invalid)",
			input: "1.2.3.4:80",
			err:   "should not include port",
		},
		{
			name:  "[IPv6] with brackets (invalid)",
			input: "[2001:db8::5]",
			err:   "should not include brackets",
		},
		{
			name:  "[IPv6]:port (invalid)",
			input: "[2001:db8::5]:80",
			err:   "should not include brackets",
		},
		{
			name:  "ipv6 with zone (invalid)",
			input: "1234::abcd%eth0",
			err:   "zone value is not allowed",
		},
		{
			name:  "leading whitespace (invalid)",
			input: " 1.2.3.4",
			err:   "should not include whitespace",
		},
		{
			name:  "trailing whitespace (invalid)",
			input: "1.2.3.4\n",
			err:   "should not include whitespace",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ip, err := ParseValidIP(tc.input)
			if !tc.valid {
				if err == nil {
					t.Errorf("expected %q to not parse with ParseValidIP but got %q", tc.input, ip.String())
				}
			} else if tc.err != "" {
				if err == nil || !strings.Contains(err.Error(), tc.err) {
					t.Errorf("expected %q to return error containing %q but got %q, %v", tc.input, tc.err, ip.String(), err)
				}
			} else if err != nil {
				t.Errorf("expected %q to parse with ParseValidIP but got %v", tc.input, err)
			} else if ip != tc.output {
				t.Errorf("expected %q to parse to %q with ParseValidIP but got %q", tc.input, tc.output.String(), ip.String())
			}

			ip, err = ParseIP(tc.input)
			if !tc.output.IsValid() {
				if err == nil {
					t.Errorf("expected %q to not parse with ParseIP but got %q", tc.input, ip.String())
				}
			} else if err != nil {
				t.Errorf("expected %q to parse with ParseIP but got %v", tc.input, err)
			} else if ip != tc.output {
				t.Errorf("expected %q to parse to %q with ParseIP but got %q", tc.input, tc.output.String(), ip.String())
			}

			ip, err = ParseCanonicalIP(tc.input)
			if tc.output.String() != tc.input || !tc.valid {
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
