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
	"strings"
	"testing"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	kubeapiserveradmission "k8s.io/apiserver/pkg/admission"
	genericoptions "k8s.io/apiserver/pkg/server/options"
	basemetrics "k8s.io/component-base/metrics"
	kubeoptions "k8s.io/kubernetes/pkg/kubeapiserver/options"
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
			err := validateServiceNodePort(tc.options)
			if err != nil && !tc.expectErrors {
				t.Errorf("expected no errors, error found %+v", err)
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
		ServiceNodePortRange:      portRange,
		KubernetesServiceNodePort: kubernetesServiceNodePort,
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

func TestValidateAPIPriorityAndFairness(t *testing.T) {
	const conflict = "conflicts with --enable-priority-and-fairness=true and --feature-gates=APIPriorityAndFairness=true"
	tests := []struct {
		runtimeConfig    string
		errShouldContain string
	}{
		{
			runtimeConfig:    "api/all=false",
			errShouldContain: conflict,
		},
		{
			runtimeConfig:    "api/beta=false",
			errShouldContain: conflict,
		},
		{
			runtimeConfig:    "flowcontrol.apiserver.k8s.io/v1beta1=false",
			errShouldContain: "",
		},
		{
			runtimeConfig:    "flowcontrol.apiserver.k8s.io/v1beta2=false",
			errShouldContain: "",
		},
		{
			runtimeConfig:    "flowcontrol.apiserver.k8s.io/v1beta3=false",
			errShouldContain: conflict,
		},
		{
			runtimeConfig:    "flowcontrol.apiserver.k8s.io/v1beta3=true",
			errShouldContain: "",
		},
	}

	for _, test := range tests {
		t.Run(test.runtimeConfig, func(t *testing.T) {
			options := &ServerRunOptions{
				GenericServerRunOptions: &genericoptions.ServerRunOptions{
					EnablePriorityAndFairness: true,
				},
				APIEnablement: genericoptions.NewAPIEnablementOptions(),
			}
			options.APIEnablement.RuntimeConfig.Set(test.runtimeConfig)

			var errMessageGot string
			if errs := validateAPIPriorityAndFairness(options); len(errs) > 0 {
				errMessageGot = errs[0].Error()
			}
			if !strings.Contains(errMessageGot, test.errShouldContain) {
				t.Errorf("Expected error message to contain: %q, but got: %q", test.errShouldContain, errMessageGot)
			}
		})
	}
}

func TestValidateServerRunOptions(t *testing.T) {
	cidrOpts := makeOptionsWithCIDRs("10.0.0.0/16", "3000::/64")
	nodePortOpts := makeOptionsWithPort(-1, 30065, 1)

	testCases := []struct {
		name         string
		options      *ServerRunOptions
		expectErrors bool
	}{
		{
			name:         "validate master count equal 0",
			expectErrors: true,
			options: &ServerRunOptions{
				MasterCount:             0,
				GenericServerRunOptions: &genericoptions.ServerRunOptions{},
				Etcd:                    &genericoptions.EtcdOptions{},
				SecureServing:           &genericoptions.SecureServingOptionsWithLoopback{},
				Audit:                   &genericoptions.AuditOptions{},
				Admission: &kubeoptions.AdmissionOptions{
					GenericAdmission: &genericoptions.AdmissionOptions{
						EnablePlugins: []string{"foo"},
						Plugins:       kubeapiserveradmission.NewPlugins(),
					},
					PluginNames: []string{"foo"},
				},
				Authentication: &kubeoptions.BuiltInAuthenticationOptions{
					APIAudiences: []string{"bar"},
					ServiceAccounts: &kubeoptions.ServiceAccountAuthenticationOptions{
						Issuers: []string{"baz"},
					},
				},
				Authorization:                  &kubeoptions.BuiltInAuthorizationOptions{},
				APIEnablement:                  genericoptions.NewAPIEnablementOptions(),
				Metrics:                        &basemetrics.Options{},
				ServiceClusterIPRanges:         cidrOpts.ServiceClusterIPRanges,
				PrimaryServiceClusterIPRange:   cidrOpts.PrimaryServiceClusterIPRange,
				SecondaryServiceClusterIPRange: cidrOpts.SecondaryServiceClusterIPRange,
				ServiceNodePortRange:           nodePortOpts.ServiceNodePortRange,
				KubernetesServiceNodePort:      nodePortOpts.KubernetesServiceNodePort,
				ServiceAccountSigningKeyFile:   "",
			},
		},
		{
			name:         "validate token request enable not attempted",
			expectErrors: true,
			options: &ServerRunOptions{
				MasterCount:             1,
				GenericServerRunOptions: &genericoptions.ServerRunOptions{},
				Etcd:                    &genericoptions.EtcdOptions{},
				SecureServing:           &genericoptions.SecureServingOptionsWithLoopback{},
				Audit:                   &genericoptions.AuditOptions{},
				Admission: &kubeoptions.AdmissionOptions{
					GenericAdmission: &genericoptions.AdmissionOptions{
						EnablePlugins: []string{""},
						Plugins:       kubeapiserveradmission.NewPlugins(),
					},
					PluginNames: []string{""},
				},
				Authentication: &kubeoptions.BuiltInAuthenticationOptions{
					ServiceAccounts: &kubeoptions.ServiceAccountAuthenticationOptions{},
				},
				Authorization:                  &kubeoptions.BuiltInAuthorizationOptions{},
				APIEnablement:                  genericoptions.NewAPIEnablementOptions(),
				Metrics:                        &basemetrics.Options{},
				ServiceClusterIPRanges:         cidrOpts.ServiceClusterIPRanges,
				PrimaryServiceClusterIPRange:   cidrOpts.PrimaryServiceClusterIPRange,
				SecondaryServiceClusterIPRange: cidrOpts.SecondaryServiceClusterIPRange,
				ServiceNodePortRange:           nodePortOpts.ServiceNodePortRange,
				KubernetesServiceNodePort:      nodePortOpts.KubernetesServiceNodePort,
				ServiceAccountSigningKeyFile:   "",
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := tc.options.Validate()
			if len(errs) > 0 && !tc.expectErrors {
				t.Errorf("expected no errors, errors found %+v", errs)
			}

			if len(errs) == 0 && tc.expectErrors {
				t.Errorf("expected errors, no errors found")
			}
		})
	}
}
