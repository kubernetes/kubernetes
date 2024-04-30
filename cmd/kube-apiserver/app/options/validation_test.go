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

	utilnet "k8s.io/apimachinery/pkg/util/net"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	netutils "k8s.io/utils/net"

	"k8s.io/kubernetes/pkg/features"
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
		Extra: Extra{
			ServiceClusterIPRanges:         value,
			PrimaryServiceClusterIPRange:   primaryCIDR,
			SecondaryServiceClusterIPRange: secondaryCIDR,
		},
	}
}

func TestClusterServiceIPRange(t *testing.T) {
	testCases := []struct {
		name         string
		options      *ServerRunOptions
		expectErrors bool
		gate         bool
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
			name:         "service cidr IPv4 is too big but gate enbled",
			expectErrors: false,
			options:      makeOptionsWithCIDRs("10.0.0.0/8", ""),
			gate:         true,
		},
		{
			name:         "service cidr IPv6 is too big but gate enbled",
			expectErrors: false,
			options:      makeOptionsWithCIDRs("2001:db8::/64", ""),
			gate:         true,
		},
		{
			name:         "service cidr IPv6 is too big and gate enbled",
			expectErrors: false,
			options:      makeOptionsWithCIDRs("2001:db8::/12", ""),
			gate:         true,
		},
		{
			name:         "dual-stack secondary cidr too big",
			expectErrors: true,
			options:      makeOptionsWithCIDRs("10.0.0.0/16", "3000::/64"),
		},
		{
			name:         "dual-stack secondary cidr too big gate enabled",
			expectErrors: false,
			options:      makeOptionsWithCIDRs("10.0.0.0/16", "3000::/48"),
			gate:         true,
		},
		{
			name:         "more than two entries",
			expectErrors: true,
			options:      makeOptionsWithCIDRs("10.0.0.0/16,244.0.0.0/16", "3000::/108"),
		},
		/* success cases */
		{
			name:         "valid primary",
			expectErrors: false,
			options:      makeOptionsWithCIDRs("10.0.0.0/16", ""),
		},
		{
			name:         "valid primary, class E range",
			expectErrors: false,
			options:      makeOptionsWithCIDRs("244.0.0.0/16", ""),
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
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MultiCIDRServiceAllocator, tc.gate)

			errs := validateClusterIPFlags(tc.options.Extra)
			if len(errs) > 0 && !tc.expectErrors {
				t.Errorf("expected no errors, errors found %+v", errs)
			}

			if len(errs) == 0 && tc.expectErrors {
				t.Errorf("expected errors, no errors found")
			}
		})
	}
}

func TestValidatePublicIPServiceClusterIPRangeIPFamilies(t *testing.T) {
	_, ipv4cidr, err := netutils.ParseCIDRSloppy("192.168.0.0/24")
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}

	_, ipv6cidr, err := netutils.ParseCIDRSloppy("2001:db8::/112")
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}

	ipv4address := netutils.ParseIPSloppy("192.168.1.1")
	ipv6address := netutils.ParseIPSloppy("2001:db8::1")

	tests := []struct {
		name    string
		generic apiserveroptions.ServerRunOptions
		extra   Extra
		wantErr bool
	}{
		{
			name: "master endpoint reconciler - IPv4 families",
			extra: Extra{
				EndpointReconcilerType:       "master-count",
				PrimaryServiceClusterIPRange: *ipv4cidr,
			},
			generic: apiserveroptions.ServerRunOptions{
				AdvertiseAddress: ipv4address,
			},
			wantErr: false,
		},
		{
			name: "master endpoint reconciler - IPv6 families",
			extra: Extra{
				EndpointReconcilerType:       "master-count",
				PrimaryServiceClusterIPRange: *ipv6cidr,
			},
			generic: apiserveroptions.ServerRunOptions{
				AdvertiseAddress: ipv6address,
			},
			wantErr: false,
		},
		{
			name: "master endpoint reconciler - wrong IP families",
			extra: Extra{
				EndpointReconcilerType:       "master-count",
				PrimaryServiceClusterIPRange: *ipv4cidr,
			},
			generic: apiserveroptions.ServerRunOptions{
				AdvertiseAddress: ipv6address,
			},
			wantErr: true,
		},
		{
			name: "master endpoint reconciler - wrong IP families",
			extra: Extra{
				EndpointReconcilerType:       "master-count",
				PrimaryServiceClusterIPRange: *ipv6cidr,
			},
			generic: apiserveroptions.ServerRunOptions{
				AdvertiseAddress: ipv4address,
			},
			wantErr: true,
		},
		{
			name: "lease endpoint reconciler - IPv4 families",
			extra: Extra{
				EndpointReconcilerType:       "lease",
				PrimaryServiceClusterIPRange: *ipv4cidr,
			},
			generic: apiserveroptions.ServerRunOptions{
				AdvertiseAddress: ipv4address,
			},
			wantErr: false,
		},
		{
			name: "lease endpoint reconciler - IPv6 families",
			extra: Extra{
				EndpointReconcilerType:       "lease",
				PrimaryServiceClusterIPRange: *ipv6cidr,
			},
			generic: apiserveroptions.ServerRunOptions{
				AdvertiseAddress: ipv6address,
			},
			wantErr: false,
		},
		{
			name: "lease endpoint reconciler - wrong IP families",
			extra: Extra{
				EndpointReconcilerType:       "lease",
				PrimaryServiceClusterIPRange: *ipv4cidr,
			},
			generic: apiserveroptions.ServerRunOptions{
				AdvertiseAddress: ipv6address,
			},
			wantErr: true,
		},
		{
			name: "lease endpoint reconciler - wrong IP families",
			extra: Extra{
				EndpointReconcilerType:       "lease",
				PrimaryServiceClusterIPRange: *ipv6cidr,
			},
			generic: apiserveroptions.ServerRunOptions{
				AdvertiseAddress: ipv4address,
			},
			wantErr: true,
		},
		{
			name: "none endpoint reconciler - wrong IP families",
			extra: Extra{
				EndpointReconcilerType:       "none",
				PrimaryServiceClusterIPRange: *ipv4cidr,
			},
			generic: apiserveroptions.ServerRunOptions{
				AdvertiseAddress: ipv6address,
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := validatePublicIPServiceClusterIPRangeIPFamilies(tt.extra, tt.generic)
			if (len(errs) > 0) != tt.wantErr {
				t.Fatalf("completedConfig.New() errors = %+v, wantErr %v", errs, tt.wantErr)
			}
		})
	}
}

func getIPnetFromCIDR(cidr string) *net.IPNet {
	_, ipnet, _ := netutils.ParseCIDRSloppy(cidr)
	return ipnet
}

func TestValidateServiceNodePort(t *testing.T) {
	testCases := []struct {
		name         string
		options      *ServerRunOptions
		expectErrors bool
	}{
		{
			name:         "validate port less than 0",
			options:      makeOptionsWithPort(-1, 30065, 1),
			expectErrors: true,
		},
		{
			name:         "validate port more than 65535",
			options:      makeOptionsWithPort(65536, 30065, 1),
			expectErrors: true,
		},
		{
			name:         "validate port equal 0",
			options:      makeOptionsWithPort(0, 0, 1),
			expectErrors: true,
		},
		{
			name:         "validate port less than base",
			options:      makeOptionsWithPort(30064, 30065, 1),
			expectErrors: true,
		},
		{
			name:         "validate port minus base more than size",
			options:      makeOptionsWithPort(30067, 30065, 1),
			expectErrors: true,
		},
		{
			name:         "validate success",
			options:      makeOptionsWithPort(30067, 30065, 5),
			expectErrors: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := validateServiceNodePort(tc.options.Extra)
			if errs != nil && !tc.expectErrors {
				t.Errorf("expected no errors, error found %+v", errs)
			}
		})
	}
}

func makeOptionsWithPort(kubernetesServiceNodePort int, base int, size int) *ServerRunOptions {
	var portRange = utilnet.PortRange{
		Base: base,
		Size: size,
	}
	return &ServerRunOptions{
		Extra: Extra{
			ServiceNodePortRange:      portRange,
			KubernetesServiceNodePort: kubernetesServiceNodePort,
		},
	}
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
			name:                 "ipv4 cidr too big",
			cidr:                 *getIPnetFromCIDR("10.92.0.0/8"),
			maxCIDRBits:          20,
			cidrFlag:             "--service-cluster-ip-range",
			expectedErrorMessage: "specified --service-cluster-ip-range is too large; for 32-bit addresses, the mask must be >= 12",
			expectErrors:         true,
		},
		{
			name:                 "ipv6 cidr too big",
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
