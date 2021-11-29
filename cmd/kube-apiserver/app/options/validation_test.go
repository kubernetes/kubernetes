/*
Copyright 2019 The Kubernetes Authors.

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

package options

import (
	"net"
	"testing"

	netutils "k8s.io/utils/net"
)

func makeOptionsWithCIDRs(serviceCIDR string, secondaryServiceCIDR string) *ServerRunOptions {
	value := serviceCIDR
	if len(secondaryServiceCIDR) > 0 {
		value = value + "," + secondaryServiceCIDR
	}

	var primaryCIDR, secondaryCIDR net.IPNet
	if len(serviceCIDR) > 0 {
		_, cidr, _ := netutils.ParseCIDRSloppy(serviceCIDR)
		if cidr != nil {
			primaryCIDR = *(cidr)
		}
	}

	if len(secondaryServiceCIDR) > 0 {
		_, cidr, _ := netutils.ParseCIDRSloppy(secondaryServiceCIDR)
		if cidr != nil {
			secondaryCIDR = *(cidr)
		}
	}
	return &ServerRunOptions{
		ServiceClusterIPRanges:         value,
		PrimaryServiceClusterIPRange:   primaryCIDR,
		SecondaryServiceClusterIPRange: secondaryCIDR,
	}
}

func TestClusterServiceIPRange(t *testing.T) {
	testCases := []struct {
		name         string
		options      *ServerRunOptions
		expectErrors bool
	}{
		{
			name:         "no service cidr",
			expectErrors: true,
			options:      makeOptionsWithCIDRs("", ""),
		},
		{
			name:         "only secondary service cidr",
			expectErrors: true,
			options:      makeOptionsWithCIDRs("", "10.0.0.0/16"),
		},
		{
			name:         "primary and secondary are provided but not dual stack v4-v4",
			expectErrors: true,
			options:      makeOptionsWithCIDRs("10.0.0.0/16", "11.0.0.0/16"),
		},
		{
			name:         "primary and secondary are provided but not dual stack v6-v6",
			expectErrors: true,
			options:      makeOptionsWithCIDRs("2000::/108", "3000::/108"),
		},
		{
			name:         "service cidr is too big",
			expectErrors: true,
			options:      makeOptionsWithCIDRs("10.0.0.0/8", ""),
		},
		{
			name:         "dual-stack secondary cidr too big",
			expectErrors: true,
			options:      makeOptionsWithCIDRs("10.0.0.0/16", "3000::/64"),
		},

		/* success cases */
		{
			name:         "valid primary",
			expectErrors: false,
			options:      makeOptionsWithCIDRs("10.0.0.0/16", ""),
		},
		{
			name:         "valid v4-v6 dual stack",
			expectErrors: false,
			options:      makeOptionsWithCIDRs("10.0.0.0/16", "3000::/108"),
		},
		{
			name:         "valid v6-v4 dual stack",
			expectErrors: false,
			options:      makeOptionsWithCIDRs("3000::/108", "10.0.0.0/16"),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := validateClusterIPFlags(tc.options)
			if len(errs) > 0 && !tc.expectErrors {
				t.Errorf("expected no errors, errors found %+v", errs)
			}

			if len(errs) == 0 && tc.expectErrors {
				t.Errorf("expected errors, no errors found")
			}
		})
	}
}

func getIPnetFromCIDR(cidr string) *net.IPNet {
	_, ipnet, _ := netutils.ParseCIDRSloppy(cidr)
	return ipnet
}

func TestValidateMaxCIDRRange(t *testing.T) {
	testCases := []struct {
		// tc.cidr, tc.maxCIDRBits, tc.cidrFlag) tc.expectedErrorMessage
		name                 string
		cidr                 net.IPNet
		maxCIDRBits          int
		cidrFlag             string
		expectedErrorMessage string
		expectErrors         bool
	}{
		{
			name:                 "valid ipv4 cidr",
			cidr:                 *getIPnetFromCIDR("10.92.0.0/12"),
			maxCIDRBits:          20,
			cidrFlag:             "--service-cluster-ip-range",
			expectedErrorMessage: "",
			expectErrors:         false,
		},
		{
			name:                 "valid ipv6 cidr",
			cidr:                 *getIPnetFromCIDR("3000::/108"),
			maxCIDRBits:          20,
			cidrFlag:             "--service-cluster-ip-range",
			expectedErrorMessage: "",
			expectErrors:         false,
		},
		{
			name:                 "ipv4 cidr to big",
			cidr:                 *getIPnetFromCIDR("10.92.0.0/8"),
			maxCIDRBits:          20,
			cidrFlag:             "--service-cluster-ip-range",
			expectedErrorMessage: "specified --service-cluster-ip-range is too large; for 32-bit addresses, the mask must be >= 12",
			expectErrors:         true,
		},
		{
			name:                 "ipv6 cidr to big",
			cidr:                 *getIPnetFromCIDR("3000::/64"),
			maxCIDRBits:          20,
			cidrFlag:             "--service-cluster-ip-range",
			expectedErrorMessage: "specified --service-cluster-ip-range is too large; for 128-bit addresses, the mask must be >= 108",
			expectErrors:         true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateMaxCIDRRange(tc.cidr, tc.maxCIDRBits, tc.cidrFlag)
			if err != nil && !tc.expectErrors {
				t.Errorf("expected no errors, error found %+v", err)
			}

			if err == nil && tc.expectErrors {
				t.Errorf("expected errors, no errors found")
			}

			if err != nil && tc.expectErrors && err.Error() != tc.expectedErrorMessage {
				t.Errorf("Expected error message: \"%s\"\nGot: \"%s\"", tc.expectedErrorMessage, err.Error())
			}
		})
	}
}
