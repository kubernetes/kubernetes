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

package validation

import (
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestIsValidIP(t *testing.T) {
	testCases := []struct {
		name string
		in   string

		err             string
		legacyErr       string
		legacyStrictErr string
	}{
		// GOOD VALUES
		{
			name: "ipv4",
			in:   "1.2.3.4",
		},
		{
			name: "ipv4, all zeros",
			in:   "0.0.0.0",
		},
		{
			name: "ipv4, max",
			in:   "255.255.255.255",
		},
		{
			name: "ipv6",
			in:   "1234::abcd",
		},
		{
			name: "ipv6, all zeros, collapsed",
			in:   "::",
		},
		{
			name: "ipv6, max",
			in:   "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
		},

		// NON-CANONICAL VALUES
		{
			name: "ipv6, all zeros, expanded (non-canonical)",
			in:   "0:0:0:0:0:0:0:0",

			err: `must be in canonical form ("::")`,
		},
		{
			name: "ipv6, leading 0s (non-canonical)",
			in:   "0001:002:03:4::",

			err: `must be in canonical form ("1:2:3:4::")`,
		},
		{
			name: "ipv6, capital letters (non-canonical)",
			in:   "1234::ABCD",

			err: `must be in canonical form ("1234::abcd")`,
		},

		// GOOD WITH LEGACY VALIDATION, BAD WITH STRICT VALIDATION
		{
			name: "ipv4 with leading 0s",
			in:   "1.1.1.01",

			err:             "must not have leading 0s",
			legacyErr:       "",
			legacyStrictErr: "must not have leading 0s",
		},
		{
			name: "ipv4-in-ipv6 value",
			in:   "::ffff:1.1.1.1",

			err:             "must not be an IPv4-mapped IPv6 address",
			legacyErr:       "",
			legacyStrictErr: "must not be an IPv4-mapped IPv6 address",
		},

		// BAD VALUES
		{
			name: "empty string",
			in:   "",

			err:             "must be a valid IP address",
			legacyErr:       "must be a valid IP address",
			legacyStrictErr: "must be a valid IP address",
		},
		{
			name: "junk",
			in:   "aaaaaaa",

			err:             "must be a valid IP address",
			legacyErr:       "must be a valid IP address",
			legacyStrictErr: "must be a valid IP address",
		},
		{
			name: "domain name",
			in:   "myhost.mydomain",

			err:             "must be a valid IP address",
			legacyErr:       "must be a valid IP address",
			legacyStrictErr: "must be a valid IP address",
		},
		{
			name: "cidr",
			in:   "1.2.3.0/24",

			err:             "must be a valid IP address",
			legacyErr:       "must be a valid IP address",
			legacyStrictErr: "must be a valid IP address",
		},
		{
			name: "ipv4 with out-of-range octets",
			in:   "1.2.3.400",

			err:             "must be a valid IP address",
			legacyErr:       "must be a valid IP address",
			legacyStrictErr: "must be a valid IP address",
		},
		{
			name: "ipv4 with negative octets",
			in:   "-1.0.0.0",

			err:             "must be a valid IP address",
			legacyErr:       "must be a valid IP address",
			legacyStrictErr: "must be a valid IP address",
		},
		{
			name: "ipv6 with out-of-range segment",
			in:   "2001:db8::10005",

			err:             "must be a valid IP address",
			legacyErr:       "must be a valid IP address",
			legacyStrictErr: "must be a valid IP address",
		},
		{
			name: "ipv4:port",
			in:   "1.2.3.4:80",

			err:             "must be a valid IP address",
			legacyErr:       "must be a valid IP address",
			legacyStrictErr: "must be a valid IP address",
		},
		{
			name: "ipv6 with brackets",
			in:   "[2001:db8::1]",

			err:             "must be a valid IP address",
			legacyErr:       "must be a valid IP address",
			legacyStrictErr: "must be a valid IP address",
		},
		{
			name: "[ipv6]:port",
			in:   "[2001:db8::1]:80",

			err:             "must be a valid IP address",
			legacyErr:       "must be a valid IP address",
			legacyStrictErr: "must be a valid IP address",
		},
		{
			name: "host:port",
			in:   "example.com:80",

			err:             "must be a valid IP address",
			legacyErr:       "must be a valid IP address",
			legacyStrictErr: "must be a valid IP address",
		},
		{
			name: "ipv6 with zone",
			in:   "1234::abcd%eth0",

			err:             "must be a valid IP address",
			legacyErr:       "must be a valid IP address",
			legacyStrictErr: "must be a valid IP address",
		},
		{
			name: "ipv4 with zone",
			in:   "169.254.0.0%eth0",

			err:             "must be a valid IP address",
			legacyErr:       "must be a valid IP address",
			legacyStrictErr: "must be a valid IP address",
		},
	}

	var badIPs []string
	for _, tc := range testCases {
		if tc.legacyStrictErr != "" {
			badIPs = append(badIPs, tc.in)
		}
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := IsValidIP(field.NewPath(""), tc.in)
			if tc.err == "" {
				if len(errs) != 0 {
					t.Errorf("expected %q to be valid but got: %v", tc.in, errs)
				}
			} else {
				if len(errs) != 1 {
					t.Errorf("expected %q to have 1 error but got: %v", tc.in, errs)
				} else if !strings.Contains(errs[0].Detail, tc.err) {
					t.Errorf("expected error for %q to contain %q but got: %q", tc.in, tc.err, errs[0].Detail)
				}
			}

			errs = IsValidIPForLegacyField(field.NewPath(""), tc.in, false, nil)
			if tc.legacyErr == "" {
				if len(errs) != 0 {
					t.Errorf("expected %q to be valid according to IsValidIPForLegacyField but got: %v", tc.in, errs)
				}
			} else {
				if len(errs) != 1 {
					t.Errorf("expected %q to have 1 error from IsValidIPForLegacyField but got: %v", tc.in, errs)
				} else if !strings.Contains(errs[0].Detail, tc.legacyErr) {
					t.Errorf("expected error from IsValidIPForLegacyField for %q to contain %q but got: %q", tc.in, tc.legacyErr, errs[0].Detail)
				}
			}

			errs = IsValidIPForLegacyField(field.NewPath(""), tc.in, true, nil)
			if tc.legacyStrictErr == "" {
				if len(errs) != 0 {
					t.Errorf("expected %q to be valid according to IsValidIPForLegacyField with strict validation, but got: %v", tc.in, errs)
				}
			} else {
				if len(errs) != 1 {
					t.Errorf("expected %q to have 1 error from IsValidIPForLegacyField with strict validation, but got: %v", tc.in, errs)
				} else if !strings.Contains(errs[0].Detail, tc.legacyStrictErr) {
					t.Errorf("expected error from IsValidIPForLegacyField with strict validation for %q to contain %q but got: %q", tc.in, tc.legacyStrictErr, errs[0].Detail)
				}
			}

			errs = IsValidIPForLegacyField(field.NewPath(""), tc.in, true, badIPs)
			if len(errs) != 0 {
				t.Errorf("expected %q to be accepted when using validOldIPs, but got: %v", tc.in, errs)
			}
		})
	}
}

func TestGetWarningsForIP(t *testing.T) {
	tests := []struct {
		name      string
		fieldPath *field.Path
		address   string
		want      []string
	}{
		{
			name:      "IPv4 No failures",
			address:   "192.12.2.2",
			fieldPath: field.NewPath("spec").Child("clusterIPs").Index(0),
			want:      nil,
		},
		{
			name:      "IPv6 No failures",
			address:   "2001:db8::2",
			fieldPath: field.NewPath("spec").Child("clusterIPs").Index(0),
			want:      nil,
		},
		{
			name:      "IPv4 with leading zeros",
			address:   "192.012.2.2",
			fieldPath: field.NewPath("spec").Child("clusterIPs").Index(0),
			want: []string{
				`spec.clusterIPs[0]: non-standard IP address "192.012.2.2" will be considered invalid in a future Kubernetes release: use "192.12.2.2"`,
			},
		},
		{
			name:      "IPv4-mapped IPv6",
			address:   "::ffff:192.12.2.2",
			fieldPath: field.NewPath("spec").Child("clusterIPs").Index(0),
			want: []string{
				`spec.clusterIPs[0]: non-standard IP address "::ffff:192.12.2.2" will be considered invalid in a future Kubernetes release: use "192.12.2.2"`,
			},
		},
		{
			name:      "IPv6 non-canonical format",
			address:   "2001:db8:0:0::2",
			fieldPath: field.NewPath("spec").Child("loadBalancerIP"),
			want: []string{
				`spec.loadBalancerIP: IPv6 address "2001:db8:0:0::2" should be in RFC 5952 canonical format ("2001:db8::2")`,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetWarningsForIP(tt.fieldPath, tt.address); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getWarningsForIP() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsValidCIDR(t *testing.T) {
	testCases := []struct {
		name string
		in   string

		err             string
		legacyErr       string
		legacyStrictErr string
	}{
		// GOOD VALUES
		{
			name: "ipv4",
			in:   "1.0.0.0/8",
		},
		{
			name: "ipv4, all IPs",
			in:   "0.0.0.0/0",
		},
		{
			name: "ipv4, single IP",
			in:   "1.1.1.1/32",
		},
		{
			name: "ipv6",
			in:   "2001:4860:4860::/48",
		},
		{
			name: "ipv6, all IPs",
			in:   "::/0",
		},
		{
			name: "ipv6, single IP",
			in:   "::1/128",
		},

		// NON-CANONICAL VALUES
		{
			name: "ipv6, extra 0s (non-canonical)",
			in:   "2a00:79e0:2:0::/64",

			err: `must be in canonical form ("2a00:79e0:2::/64")`,
		},
		{
			name: "ipv6, capital letters (non-canonical)",
			in:   "2001:DB8::/64",

			err: `must be in canonical form ("2001:db8::/64")`,
		},

		// GOOD WITH LEGACY VALIDATION, BAD WITH STRICT VALIDATION
		{
			name: "ipv4 with leading 0s",
			in:   "1.1.01.0/24",

			err:             "must not have leading 0s in IP",
			legacyErr:       "",
			legacyStrictErr: "must not have leading 0s in IP",
		},
		{
			name: "ipv4-in-ipv6 with ipv4-sized prefix",
			in:   "::ffff:1.1.1.0/24",

			err:             "must not have an IPv4-mapped IPv6 address",
			legacyErr:       "",
			legacyStrictErr: "must not have an IPv4-mapped IPv6 address",
		},
		{
			name: "ipv4-in-ipv6 with ipv6-sized prefix",
			in:   "::ffff:1.1.1.0/120",

			err:             "must not have an IPv4-mapped IPv6 address",
			legacyErr:       "",
			legacyStrictErr: "must not have an IPv4-mapped IPv6 address",
		},
		{
			name: "ipv4 ifaddr",
			in:   "1.2.3.4/24",

			err:             "must not have bits set beyond the prefix length",
			legacyErr:       "",
			legacyStrictErr: "must not have bits set beyond the prefix length",
		},
		{
			name: "ipv6 ifaddr",
			in:   "2001:db8::1/64",

			err:             "must not have bits set beyond the prefix length",
			legacyErr:       "",
			legacyStrictErr: "must not have bits set beyond the prefix length",
		},
		{
			name: "prefix length with leading 0s",
			in:   "192.168.0.0/016",

			err:             "must not have leading 0s",
			legacyErr:       "",
			legacyStrictErr: "must not have leading 0s",
		},

		// BAD VALUES
		{
			name: "empty string",
			in:   "",

			err:             "must be a valid CIDR value",
			legacyErr:       "must be a valid CIDR value",
			legacyStrictErr: "must be a valid CIDR value",
		},
		{
			name: "junk",
			in:   "aaaaaaa",

			err:             "must be a valid CIDR value",
			legacyErr:       "must be a valid CIDR value",
			legacyStrictErr: "must be a valid CIDR value",
		},
		{
			name: "IP address",
			in:   "1.2.3.4",

			err:             "must be a valid CIDR value",
			legacyErr:       "must be a valid CIDR value",
			legacyStrictErr: "must be a valid CIDR value",
		},
		{
			name: "partial URL",
			in:   "192.168.0.1/healthz",

			err:             "must be a valid CIDR value",
			legacyErr:       "must be a valid CIDR value",
			legacyStrictErr: "must be a valid CIDR value",
		},
		{
			name: "partial URL 2",
			in:   "192.168.0.1/0/99",

			err:             "must be a valid CIDR value",
			legacyErr:       "must be a valid CIDR value",
			legacyStrictErr: "must be a valid CIDR value",
		},
		{
			name: "negative prefix length",
			in:   "192.168.0.0/-16",

			err:             "must be a valid CIDR value",
			legacyErr:       "must be a valid CIDR value",
			legacyStrictErr: "must be a valid CIDR value",
		},
		{
			name: "prefix length with sign",
			in:   "192.168.0.0/+16",

			err:             "must be a valid CIDR value",
			legacyErr:       "must be a valid CIDR value",
			legacyStrictErr: "must be a valid CIDR value",
		},
	}

	var badCIDRs []string
	for _, tc := range testCases {
		if tc.legacyStrictErr != "" {
			badCIDRs = append(badCIDRs, tc.in)
		}
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := IsValidCIDR(field.NewPath(""), tc.in)
			if tc.err == "" {
				if len(errs) != 0 {
					t.Errorf("expected %q to be valid but got: %v", tc.in, errs)
				}
			} else {
				if len(errs) != 1 {
					t.Errorf("expected %q to have 1 error but got: %v", tc.in, errs)
				} else if !strings.Contains(errs[0].Detail, tc.err) {
					t.Errorf("expected error for %q to contain %q but got: %q", tc.in, tc.err, errs[0].Detail)
				}
			}

			errs = IsValidCIDRForLegacyField(field.NewPath(""), tc.in, false, nil)
			if tc.legacyErr == "" {
				if len(errs) != 0 {
					t.Errorf("expected %q to be valid according to IsValidCIDRForLegacyField but got: %v", tc.in, errs)
				}
			} else {
				if len(errs) != 1 {
					t.Errorf("expected %q to have 1 error from IsValidCIDRForLegacyField but got: %v", tc.in, errs)
				} else if !strings.Contains(errs[0].Detail, tc.legacyErr) {
					t.Errorf("expected error for %q from IsValidCIDRForLegacyField to contain %q but got: %q", tc.in, tc.legacyErr, errs[0].Detail)
				}
			}

			errs = IsValidCIDRForLegacyField(field.NewPath(""), tc.in, true, nil)
			if tc.legacyStrictErr == "" {
				if len(errs) != 0 {
					t.Errorf("expected %q to be valid according to IsValidCIDRForLegacyField with strict validation but got: %v", tc.in, errs)
				}
			} else {
				if len(errs) != 1 {
					t.Errorf("expected %q to have 1 error from IsValidCIDRForLegacyField with strict validation but got: %v", tc.in, errs)
				} else if !strings.Contains(errs[0].Detail, tc.legacyStrictErr) {
					t.Errorf("expected error for %q from IsValidCIDRForLegacyField with strict validation to contain %q but got: %q", tc.in, tc.legacyStrictErr, errs[0].Detail)
				}
			}

			errs = IsValidCIDRForLegacyField(field.NewPath(""), tc.in, true, badCIDRs)
			if len(errs) != 0 {
				t.Errorf("expected %q to be accepted when using validOldCIDRs, but got: %v", tc.in, errs)
			}
		})
	}
}

func TestGetWarningsForCIDR(t *testing.T) {
	tests := []struct {
		name      string
		fieldPath *field.Path
		cidr      string
		want      []string
	}{
		{
			name:      "IPv4 No failures",
			cidr:      "192.12.2.0/24",
			fieldPath: field.NewPath("spec").Child("loadBalancerSourceRanges").Index(0),
			want:      nil,
		},
		{
			name:      "IPv6 No failures",
			cidr:      "2001:db8::/64",
			fieldPath: field.NewPath("spec").Child("loadBalancerSourceRanges").Index(0),
			want:      nil,
		},
		{
			name:      "IPv4 with leading zeros",
			cidr:      "192.012.2.0/24",
			fieldPath: field.NewPath("spec").Child("loadBalancerSourceRanges").Index(0),
			want: []string{
				`spec.loadBalancerSourceRanges[0]: non-standard CIDR value "192.012.2.0/24" will be considered invalid in a future Kubernetes release: use "192.12.2.0/24"`,
			},
		},
		{
			name:      "leading zeros in prefix length",
			cidr:      "192.12.2.0/024",
			fieldPath: field.NewPath("spec").Child("loadBalancerSourceRanges").Index(0),
			want: []string{
				`spec.loadBalancerSourceRanges[0]: non-standard CIDR value "192.12.2.0/024" will be considered invalid in a future Kubernetes release: use "192.12.2.0/24"`,
			},
		},
		{
			name:      "IPv4-mapped IPv6",
			cidr:      "::ffff:192.12.2.0/120",
			fieldPath: field.NewPath("spec").Child("loadBalancerSourceRanges").Index(0),
			want: []string{
				`spec.loadBalancerSourceRanges[0]: non-standard CIDR value "::ffff:192.12.2.0/120" will be considered invalid in a future Kubernetes release: use "192.12.2.0/24"`,
			},
		},
		{
			name:      "bits after prefix length",
			cidr:      "192.12.2.8/24",
			fieldPath: field.NewPath("spec").Child("loadBalancerSourceRanges").Index(0),
			want: []string{
				`spec.loadBalancerSourceRanges[0]: CIDR value "192.12.2.8/24" is ambiguous in this context (should be "192.12.2.0/24" or "192.12.2.8/32"?)`,
			},
		},
		{
			name:      "multiple problems",
			cidr:      "192.012.2.8/24",
			fieldPath: field.NewPath("spec").Child("loadBalancerSourceRanges").Index(0),
			want: []string{
				`spec.loadBalancerSourceRanges[0]: CIDR value "192.012.2.8/24" is ambiguous in this context (should be "192.12.2.0/24" or "192.12.2.8/32"?)`,
				`spec.loadBalancerSourceRanges[0]: non-standard CIDR value "192.012.2.8/24" will be considered invalid in a future Kubernetes release: use "192.12.2.0/24"`,
			},
		},
		{
			name:      "IPv6 non-canonical format",
			cidr:      "2001:db8:0:0::/64",
			fieldPath: field.NewPath("spec").Child("loadBalancerSourceRanges").Index(0),
			want: []string{
				`spec.loadBalancerSourceRanges[0]: IPv6 CIDR value "2001:db8:0:0::/64" should be in RFC 5952 canonical format ("2001:db8::/64")`,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetWarningsForCIDR(tt.fieldPath, tt.cidr); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getWarningsForCIDR() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsValidInterfaceAddress(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   string
		err  string
	}{
		// GOOD VALUES
		{
			name: "ipv4",
			in:   "1.2.3.4/24",
		},
		{
			name: "ipv4, single IP",
			in:   "1.1.1.1/32",
		},
		{
			name: "ipv6",
			in:   "2001:4860:4860::1/48",
		},
		{
			name: "ipv6, single IP",
			in:   "::1/128",
		},

		// BAD VALUES
		{
			name: "empty string",
			in:   "",
			err:  "must be a valid address in CIDR form",
		},
		{
			name: "junk",
			in:   "aaaaaaa",
			err:  "must be a valid address in CIDR form",
		},
		{
			name: "IP address",
			in:   "1.2.3.4",
			err:  "must be a valid address in CIDR form",
		},
		{
			name: "partial URL",
			in:   "192.168.0.1/healthz",
			err:  "must be a valid address in CIDR form",
		},
		{
			name: "partial URL 2",
			in:   "192.168.0.1/0/99",
			err:  "must be a valid address in CIDR form",
		},
		{
			name: "negative prefix length",
			in:   "192.168.0.0/-16",
			err:  "must be a valid address in CIDR form",
		},
		{
			name: "prefix length with sign",
			in:   "192.168.0.0/+16",
			err:  "must be a valid address in CIDR form",
		},
		{
			name: "ipv6 non-canonical",
			in:   "2001:0:0:0::0BCD/64",
			err:  `must be in canonical form ("2001::bcd/64")`,
		},
		{
			name: "ipv4 with leading 0s",
			in:   "1.1.01.002/24",
			err:  `must be in canonical form ("1.1.1.2/24")`,
		},
		{
			name: "ipv4-in-ipv6 with ipv4-sized prefix",
			in:   "::ffff:1.1.1.1/24",
			err:  `must be in canonical form ("1.1.1.1/24")`,
		},
		{
			name: "ipv4-in-ipv6 with ipv6-sized prefix",
			in:   "::ffff:1.1.1.1/120",
			err:  `must be in canonical form ("1.1.1.1/24")`,
		},
		{
			name: "prefix length with leading 0s",
			in:   "192.168.0.5/016",
			err:  `must be in canonical form ("192.168.0.5/16")`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			errs := IsValidInterfaceAddress(field.NewPath(""), tc.in)
			if tc.err == "" {
				if len(errs) != 0 {
					t.Errorf("expected %q to be valid but got: %v", tc.in, errs)
				}
			} else {
				if len(errs) != 1 {
					t.Errorf("expected %q to have 1 error but got: %v", tc.in, errs)
				} else if !strings.Contains(errs[0].Detail, tc.err) {
					t.Errorf("expected error for %q to contain %q but got: %q", tc.in, tc.err, errs[0].Detail)
				}
			}
		})
	}
}
