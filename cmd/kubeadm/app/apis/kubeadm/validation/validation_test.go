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

package validation

import (
	"io/ioutil"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/spf13/pflag"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeletconfigv1alpha1 "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/v1alpha1"
	kubeproxyconfigv1alpha1 "k8s.io/kubernetes/pkg/proxy/apis/kubeproxyconfig/v1alpha1"
	utilpointer "k8s.io/kubernetes/pkg/util/pointer"
)

func TestValidateTokenDiscovery(t *testing.T) {
	var tests = []struct {
		c        *kubeadm.NodeConfiguration
		f        *field.Path
		expected bool
	}{
		{&kubeadm.NodeConfiguration{Token: "772ef5.6b6baab1d4a0a171", DiscoveryTokenAPIServers: []string{"192.168.122.100:6443"}}, nil, true},
		{&kubeadm.NodeConfiguration{Token: ".6b6baab1d4a0a171", DiscoveryTokenAPIServers: []string{"192.168.122.100:6443"}}, nil, false},
		{&kubeadm.NodeConfiguration{Token: "772ef5.", DiscoveryTokenAPIServers: []string{"192.168.122.100:6443"}}, nil, false},
		{&kubeadm.NodeConfiguration{Token: "772ef5.6b6baab1d4a0a171", DiscoveryTokenAPIServers: []string{"2001:db8::100:6443"}}, nil, true},
		{&kubeadm.NodeConfiguration{Token: ".6b6baab1d4a0a171", DiscoveryTokenAPIServers: []string{"2001:db8::100:6443"}}, nil, false},
		{&kubeadm.NodeConfiguration{Token: "772ef5.", DiscoveryTokenAPIServers: []string{"2001:db8::100:6443"}}, nil, false},
		{&kubeadm.NodeConfiguration{Token: "abcdef.1234567890123456@foobar", DiscoveryTokenAPIServers: []string{"192.168.122.100:6443"}}, nil, false},
	}
	for _, rt := range tests {
		err := ValidateToken(rt.c.Token, rt.f).ToAggregate()
		if (err == nil) != rt.expected {
			t.Errorf(
				"failed ValidateTokenDiscovery:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(err == nil),
			)
		}
	}
}

func TestValidateAuthorizationModes(t *testing.T) {
	var tests = []struct {
		s        []string
		f        *field.Path
		expected bool
	}{
		{[]string{""}, nil, false},
		{[]string{"rBAC"}, nil, false},                               // mode not supported
		{[]string{"rBAC", "Webhook"}, nil, false},                    // mode not supported
		{[]string{"RBAC", "Webhook"}, nil, false},                    // mode Node required
		{[]string{"Node", "RBAC", "Webhook", "Webhook"}, nil, false}, // no duplicates allowed
		{[]string{"not valid"}, nil, false},                          // invalid mode
		{[]string{"Node", "RBAC"}, nil, true},                        // supported
		{[]string{"RBAC", "Node"}, nil, true},                        // supported
		{[]string{"Node", "RBAC", "Webhook", "ABAC"}, nil, true},     // supported
	}
	for _, rt := range tests {
		actual := ValidateAuthorizationModes(rt.s, rt.f)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"failed ValidateAuthorizationModes:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}

func TestValidateNodeName(t *testing.T) {
	var tests = []struct {
		s        string
		f        *field.Path
		expected bool
	}{
		{"", nil, false},                 // ok if not provided
		{"1234", nil, true},              // supported
		{"valid-nodename", nil, true},    // supported
		{"INVALID-NODENAME", nil, false}, // Upper cases is invalid
	}
	for _, rt := range tests {
		actual := ValidateNodeName(rt.s, rt.f)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"failed ValidateNodeName:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}

func TestValidateCloudProvider(t *testing.T) {
	var tests = []struct {
		s        string
		f        *field.Path
		expected bool
	}{
		{"", nil, true},      // if not provided, ok, it's optional
		{"1234", nil, false}, // not supported
		{"awws", nil, false}, // not supported
		{"aws", nil, true},   // supported
		{"gce", nil, true},   // supported
	}
	for _, rt := range tests {
		actual := ValidateCloudProvider(rt.s, rt.f)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"failed ValidateCloudProvider:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}

func TestValidateAPIServerCertSANs(t *testing.T) {
	var tests = []struct {
		sans     []string
		expected bool
	}{
		{[]string{}, true},                                                     // ok if not provided
		{[]string{"1,2,,3"}, false},                                            // not a DNS label or IP
		{[]string{"my-hostname", "???&?.garbage"}, false},                      // not valid
		{[]string{"my-hostname", "my.subdomain", "1.2.3.4"}, true},             // supported
		{[]string{"my-hostname2", "my.other.subdomain", "10.0.0.10"}, true},    // supported
		{[]string{"my-hostname", "my.subdomain", "2001:db8::4"}, true},         // supported
		{[]string{"my-hostname2", "my.other.subdomain", "2001:db8::10"}, true}, // supported
	}
	for _, rt := range tests {
		actual := ValidateAPIServerCertSANs(rt.sans, nil)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"failed ValidateAPIServerCertSANs:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}

func TestValidateIPFromString(t *testing.T) {
	var tests = []struct {
		name     string
		ip       string
		expected bool
	}{
		{"invalid missing address", "", false},
		{"invalid missing decimal points in IPv4 address", "1234", false},
		{"invalid incomplete IPv4 address", "1.2", false},
		{"invalid IPv4 CIDR provided instead of IPv4 address", "1.2.3.4/16", false},
		{"valid IPv4 address", "1.2.3.4", true},
		{"valid IPv6 address", "2001:db8::1", true},
		{"invalid IPv6 CIDR provided instead of IPv6 address", "2001:db8::1/64", false},
		{"invalid hex character in IPv6 address", "2001:xb8::", false},
		{"invalid use of colons in IPv6 address", "2001::db8::", false},
	}
	for _, rt := range tests {
		actual := ValidateIPFromString(rt.ip, nil)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"%s test case failed:\n\texpected: %t\n\t  actual: %t",
				rt.name,
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}

func TestValidateIPNetFromString(t *testing.T) {
	var tests = []struct {
		name     string
		subnet   string
		minaddrs int64
		expected bool
	}{
		{"invalid missing CIDR", "", 0, false},
		{"invalid CIDR missing decimal points in IPv4 address and / mask", "1234", 0, false},
		{"invalid CIDR use of letters instead of numbers and / mask", "abc", 0, false},
		{"invalid IPv4 address provided instead of CIDR representation", "1.2.3.4", 0, false},
		{"invalid IPv6 address provided instead of CIDR representation", "2001:db8::1", 0, false},
		{"valid, but IPv4 CIDR too small. At least 10 addresses needed", "10.0.0.16/29", 10, false},
		{"valid, but IPv6 CIDR too small. At least 10 addresses needed", "2001:db8::/125", 10, false},
		{"valid IPv4 CIDR", "10.0.0.16/12", 10, true},
		{"valid IPv6 CIDR", "2001:db8::/98", 10, true},
	}
	for _, rt := range tests {
		actual := ValidateIPNetFromString(rt.subnet, rt.minaddrs, nil)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"%s test case failed :\n\texpected: %t\n\t  actual: %t",
				rt.name,
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}

func TestValidateAPIEndpoint(t *testing.T) {
	var tests = []struct {
		name     string
		s        *kubeadm.MasterConfiguration
		expected bool
	}{
		{
			name:     "Missing configuration",
			s:        &kubeadm.MasterConfiguration{},
			expected: false,
		},
		{
			name: "Valid IPv4 address and default port",
			s: &kubeadm.MasterConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
			},
			expected: true,
		},
		{
			name: "Valid IPv6 address and port",
			s: &kubeadm.MasterConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "2001:db7::1",
					BindPort:         3446,
				},
			},
			expected: true,
		},
		{
			name: "Invalid IPv4 address",
			s: &kubeadm.MasterConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "1.2.34",
					BindPort:         6443,
				},
			},
			expected: false,
		},
		{
			name: "Invalid IPv6 address",
			s: &kubeadm.MasterConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "2001:db7:1",
					BindPort:         3446,
				},
			},
			expected: false,
		},
	}
	for _, rt := range tests {
		actual := ValidateAPIEndpoint(rt.s, nil)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"%s test case failed:\n\texpected: %t\n\t  actual: %t",
				rt.name,
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}

func TestValidateMasterConfiguration(t *testing.T) {
	nodename := "valid-nodename"
	var tests = []struct {
		name     string
		s        *kubeadm.MasterConfiguration
		expected bool
	}{
		{"invalid missing master configuration",
			&kubeadm.MasterConfiguration{}, false},
		{"invalid missing token with IPv4 service subnet",
			&kubeadm.MasterConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				AuthorizationModes: []string{"Node", "RBAC"},
				Networking: kubeadm.Networking{
					ServiceSubnet: "10.96.0.1/12",
					DNSDomain:     "cluster.local",
				},
				CertificatesDir: "/some/cert/dir",
				NodeName:        nodename,
			}, false},
		{"invalid missing token with IPv6 service subnet",
			&kubeadm.MasterConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				AuthorizationModes: []string{"Node", "RBAC"},
				Networking: kubeadm.Networking{
					ServiceSubnet: "2001:db8::1/98",
					DNSDomain:     "cluster.local",
				},
				CertificatesDir: "/some/cert/dir",
				NodeName:        nodename,
			}, false},
		{"invalid missing node name",
			&kubeadm.MasterConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				AuthorizationModes: []string{"Node", "RBAC"},
				Networking: kubeadm.Networking{
					ServiceSubnet: "10.96.0.1/12",
					DNSDomain:     "cluster.local",
				},
				CertificatesDir: "/some/other/cert/dir",
				Token:           "abcdef.0123456789abcdef",
			}, false},
		{"valid master configuration with incorrect IPv4 pod subnet",
			&kubeadm.MasterConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				AuthorizationModes: []string{"Node", "RBAC"},
				Networking: kubeadm.Networking{
					ServiceSubnet: "10.96.0.1/12",
					DNSDomain:     "cluster.local",
					PodSubnet:     "10.0.1.15",
				},
				CertificatesDir: "/some/other/cert/dir",
				Token:           "abcdef.0123456789abcdef",
				NodeName:        nodename,
			}, false},
		{"valid master configuration with IPv4 service subnet",
			&kubeadm.MasterConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				KubeProxy: kubeadm.KubeProxy{
					Config: &kubeproxyconfigv1alpha1.KubeProxyConfiguration{
						BindAddress:        "192.168.59.103",
						HealthzBindAddress: "0.0.0.0:10256",
						MetricsBindAddress: "127.0.0.1:10249",
						ClusterCIDR:        "192.168.59.0/24",
						UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
						ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
						IPTables: kubeproxyconfigv1alpha1.KubeProxyIPTablesConfiguration{
							MasqueradeAll: true,
							SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
						},
						IPVS: kubeproxyconfigv1alpha1.KubeProxyIPVSConfiguration{
							SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
						},
						Conntrack: kubeproxyconfigv1alpha1.KubeProxyConntrackConfiguration{
							Max:        utilpointer.Int32Ptr(2),
							MaxPerCore: utilpointer.Int32Ptr(1),
							Min:        utilpointer.Int32Ptr(1),
							TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
							TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
						},
					},
				},
				AuthorizationModes: []string{"Node", "RBAC"},
				Networking: kubeadm.Networking{
					ServiceSubnet: "10.96.0.1/12",
					DNSDomain:     "cluster.local",
					PodSubnet:     "10.0.1.15/16",
				},
				CertificatesDir: "/some/other/cert/dir",
				Token:           "abcdef.0123456789abcdef",
				NodeName:        nodename,
			}, true},
		{"valid master configuration using IPv6 service subnet",
			&kubeadm.MasterConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "1:2:3::4",
					BindPort:         3446,
				},
				KubeProxy: kubeadm.KubeProxy{
					Config: &kubeproxyconfigv1alpha1.KubeProxyConfiguration{
						BindAddress:        "192.168.59.103",
						HealthzBindAddress: "0.0.0.0:10256",
						MetricsBindAddress: "127.0.0.1:10249",
						ClusterCIDR:        "192.168.59.0/24",
						UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
						ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
						IPTables: kubeproxyconfigv1alpha1.KubeProxyIPTablesConfiguration{
							MasqueradeAll: true,
							SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
						},
						IPVS: kubeproxyconfigv1alpha1.KubeProxyIPVSConfiguration{
							SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
						},
						Conntrack: kubeproxyconfigv1alpha1.KubeProxyConntrackConfiguration{
							Max:        utilpointer.Int32Ptr(2),
							MaxPerCore: utilpointer.Int32Ptr(1),
							Min:        utilpointer.Int32Ptr(1),
							TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
							TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
						},
					},
				},
				AuthorizationModes: []string{"Node", "RBAC"},
				Networking: kubeadm.Networking{
					ServiceSubnet: "2001:db8::1/98",
					DNSDomain:     "cluster.local",
				},
				CertificatesDir: "/some/other/cert/dir",
				Token:           "abcdef.0123456789abcdef",
				NodeName:        nodename,
			}, true},
	}
	for _, rt := range tests {
		actual := ValidateMasterConfiguration(rt.s)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"%s test case failed:\n\texpected: %t\n\t  actual: %t",
				rt.name,
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}

func TestValidateNodeConfiguration(t *testing.T) {
	var tests = []struct {
		s        *kubeadm.NodeConfiguration
		expected bool
	}{
		{&kubeadm.NodeConfiguration{}, false},
		{&kubeadm.NodeConfiguration{
			DiscoveryFile:  "foo",
			DiscoveryToken: "abcdef.1234567890123456@foobar",
			CACertPath:     "/some/cert.crt",
		}, false},
	}
	for _, rt := range tests {
		actual := ValidateNodeConfiguration(rt.s)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"failed ValidateNodeConfiguration:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}

func TestValidateMixedArguments(t *testing.T) {
	var tests = []struct {
		args     []string
		expected bool
	}{
		// Expected to succeed, --config is mixed with skip-* flags only or no other flags
		{[]string{"--foo=bar"}, true},
		{[]string{"--config=hello"}, true},
		{[]string{"--config=hello", "--skip-preflight-checks=true"}, true},
		{[]string{"--config=hello", "--skip-token-print=true"}, true},
		{[]string{"--config=hello", "--skip-preflight-checks", "--skip-token-print"}, true},
		// Expected to fail, --config is mixed with the --foo flag
		{[]string{"--config=hello", "--skip-preflight-checks", "--foo=bar"}, false},
		{[]string{"--config=hello", "--foo=bar"}, false},
	}

	var cfgPath string
	for _, rt := range tests {
		f := pflag.NewFlagSet("test", pflag.ContinueOnError)
		if f.Parsed() {
			t.Error("f.Parse() = true before Parse")
		}
		f.String("foo", "", "flag bound to config object")
		f.Bool("skip-preflight-checks", false, "flag not bound to config object")
		f.Bool("skip-token-print", false, "flag not bound to config object")
		f.StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file")
		if err := f.Parse(rt.args); err != nil {
			t.Fatal(err)
		}

		actual := ValidateMixedArguments(f)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed ValidateMixedArguments:\n\texpected: %t\n\t  actual: %t testdata: %v",
				rt.expected,
				(actual == nil),
				rt.args,
			)
		}
	}
}

func TestValidateFeatureGates(t *testing.T) {
	type featureFlag map[string]bool
	var tests = []struct {
		featureGates featureFlag
		expected     bool
	}{
		{featureFlag{"SelfHosting": true}, true},
		{featureFlag{"SelfHosting": false}, true},
		{featureFlag{"StoreCertsInSecrets": true}, true},
		{featureFlag{"StoreCertsInSecrets": false}, true},
		{featureFlag{"Foo": true}, false},
	}
	for _, rt := range tests {
		actual := ValidateFeatureGates(rt.featureGates, nil)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"failed featureGates:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}

func TestValidateIgnorePreflightErrors(t *testing.T) {
	var tests = []struct {
		ignorePreflightErrors []string
		skipPreflightChecks   bool
		expectedLen           int
		expectedError         bool
	}{
		{[]string{}, false, 0, false},                             // empty list, no old skip-preflight-checks
		{[]string{}, true, 1, false},                              // empty list, old skip-preflight-checks
		{[]string{"check1", "check2"}, false, 2, false},           // non-duplicate
		{[]string{"check1", "check2"}, true, 3, true},             // non-duplicate, but skip-preflight-checks
		{[]string{"check1", "check2", "check1"}, false, 2, false}, // duplicates
		{[]string{"check1", "check2", "all"}, false, 3, true},     // non-duplicate, but 'all' present together wth individual checks
		{[]string{"all"}, false, 1, false},                        // skip all checks by using new flag
		{[]string{"all"}, true, 1, false},                         // skip all checks by using both old and new flags at the same time
	}
	for _, rt := range tests {
		result, err := ValidateIgnorePreflightErrors(rt.ignorePreflightErrors, rt.skipPreflightChecks)
		switch {
		case err != nil && !rt.expectedError:
			t.Errorf("ValidateIgnorePreflightErrors: unexpected error for input (%s, %v), error: %v", rt.ignorePreflightErrors, rt.skipPreflightChecks, err)
		case err == nil && rt.expectedError:
			t.Errorf("ValidateIgnorePreflightErrors: expected error for input (%s, %v) but got: %v", rt.ignorePreflightErrors, rt.skipPreflightChecks, result)
		case result.Len() != rt.expectedLen:
			t.Errorf("ValidateIgnorePreflightErrors: expected Len = %d for input (%s, %v) but got: %v, %v", rt.expectedLen, rt.ignorePreflightErrors, rt.skipPreflightChecks, result.Len(), result)
		}
	}
}

func TestValidateKubeletConfiguration(t *testing.T) {
	successCase := &kubeadm.KubeletConfiguration{
		BaseConfig: &kubeletconfigv1alpha1.KubeletConfiguration{
			CgroupsPerQOS:               utilpointer.BoolPtr(true),
			EnforceNodeAllocatable:      []string{"pods", "system-reserved", "kube-reserved"},
			SystemCgroups:               "",
			CgroupRoot:                  "",
			CAdvisorPort:                utilpointer.Int32Ptr(0),
			EventBurst:                  10,
			EventRecordQPS:              utilpointer.Int32Ptr(5),
			HealthzPort:                 utilpointer.Int32Ptr(10248),
			ImageGCHighThresholdPercent: utilpointer.Int32Ptr(85),
			ImageGCLowThresholdPercent:  utilpointer.Int32Ptr(80),
			IPTablesDropBit:             utilpointer.Int32Ptr(15),
			IPTablesMasqueradeBit:       utilpointer.Int32Ptr(14),
			KubeAPIBurst:                10,
			KubeAPIQPS:                  utilpointer.Int32Ptr(5),
			MaxOpenFiles:                1000000,
			MaxPods:                     110,
			OOMScoreAdj:                 utilpointer.Int32Ptr(-999),
			PodsPerCore:                 100,
			Port:                        65535,
			ReadOnlyPort:                utilpointer.Int32Ptr(0),
			RegistryBurst:               10,
			RegistryPullQPS:             utilpointer.Int32Ptr(5),
			HairpinMode:                 "promiscuous-bridge",
		},
	}
	if allErrors := ValidateKubeletConfiguration(successCase, nil); len(allErrors) != 0 {
		t.Errorf("failed ValidateKubeletConfiguration: expect no errors but got %v", allErrors)
	}

	errorCase := &kubeadm.KubeletConfiguration{
		BaseConfig: &kubeletconfigv1alpha1.KubeletConfiguration{
			CgroupsPerQOS:               utilpointer.BoolPtr(false),
			EnforceNodeAllocatable:      []string{"pods", "system-reserved", "kube-reserved", "illegal-key"},
			SystemCgroups:               "/",
			CgroupRoot:                  "",
			CAdvisorPort:                utilpointer.Int32Ptr(-10),
			EventBurst:                  -10,
			EventRecordQPS:              utilpointer.Int32Ptr(-10),
			HealthzPort:                 utilpointer.Int32Ptr(-10),
			ImageGCHighThresholdPercent: utilpointer.Int32Ptr(101),
			ImageGCLowThresholdPercent:  utilpointer.Int32Ptr(101),
			IPTablesDropBit:             utilpointer.Int32Ptr(-10),
			IPTablesMasqueradeBit:       utilpointer.Int32Ptr(-10),
			KubeAPIBurst:                -10,
			KubeAPIQPS:                  utilpointer.Int32Ptr(-10),
			MaxOpenFiles:                -10,
			MaxPods:                     -10,
			OOMScoreAdj:                 utilpointer.Int32Ptr(-1001),
			PodsPerCore:                 -10,
			Port:                        0,
			ReadOnlyPort:                utilpointer.Int32Ptr(-10),
			RegistryBurst:               -10,
			RegistryPullQPS:             utilpointer.Int32Ptr(-10),
		},
	}
	if allErrors := ValidateKubeletConfiguration(errorCase, nil); len(allErrors) == 0 {
		t.Errorf("failed ValidateKubeletConfiguration: expect errors but got no error")
	}
}

func TestValidateKubeProxyConfiguration(t *testing.T) {
	successCases := []kubeadm.MasterConfiguration{
		{
			KubeProxy: kubeadm.KubeProxy{
				Config: &kubeproxyconfigv1alpha1.KubeProxyConfiguration{
					BindAddress:        "192.168.59.103",
					HealthzBindAddress: "0.0.0.0:10256",
					MetricsBindAddress: "127.0.0.1:10249",
					ClusterCIDR:        "192.168.59.0/24",
					UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
					ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
					IPTables: kubeproxyconfigv1alpha1.KubeProxyIPTablesConfiguration{
						MasqueradeAll: true,
						SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
						MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
					},
					IPVS: kubeproxyconfigv1alpha1.KubeProxyIPVSConfiguration{
						SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
						MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
					},
					Conntrack: kubeproxyconfigv1alpha1.KubeProxyConntrackConfiguration{
						Max:        utilpointer.Int32Ptr(2),
						MaxPerCore: utilpointer.Int32Ptr(1),
						Min:        utilpointer.Int32Ptr(1),
						TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
						TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
					},
				},
			},
		},
	}

	for _, successCase := range successCases {
		if errs := ValidateProxy(&successCase, nil); len(errs) != 0 {
			t.Errorf("failed ValidateProxy: expect no errors but got %v", errs)
		}
	}

	errorCases := []struct {
		masterConfig kubeadm.MasterConfiguration
		msg          string
	}{
		{
			masterConfig: kubeadm.MasterConfiguration{
				KubeProxy: kubeadm.KubeProxy{
					Config: &kubeproxyconfigv1alpha1.KubeProxyConfiguration{
						// only BindAddress is invalid
						BindAddress:        "10.10.12.11:2000",
						HealthzBindAddress: "0.0.0.0:10256",
						MetricsBindAddress: "127.0.0.1:10249",
						ClusterCIDR:        "192.168.59.0/24",
						UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
						ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
						IPTables: kubeproxyconfigv1alpha1.KubeProxyIPTablesConfiguration{
							MasqueradeAll: true,
							SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
						},
						IPVS: kubeproxyconfigv1alpha1.KubeProxyIPVSConfiguration{
							SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
						},
						Conntrack: kubeproxyconfigv1alpha1.KubeProxyConntrackConfiguration{
							Max:        utilpointer.Int32Ptr(2),
							MaxPerCore: utilpointer.Int32Ptr(1),
							Min:        utilpointer.Int32Ptr(1),
							TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
							TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
						},
					},
				},
			},
			msg: "not a valid textual representation of an IP address",
		},
		{
			masterConfig: kubeadm.MasterConfiguration{
				KubeProxy: kubeadm.KubeProxy{
					Config: &kubeproxyconfigv1alpha1.KubeProxyConfiguration{
						BindAddress: "10.10.12.11",
						// only HealthzBindAddress is invalid
						HealthzBindAddress: "0.0.0.0",
						MetricsBindAddress: "127.0.0.1:10249",
						ClusterCIDR:        "192.168.59.0/24",
						UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
						ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
						IPTables: kubeproxyconfigv1alpha1.KubeProxyIPTablesConfiguration{
							MasqueradeAll: true,
							SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
						},
						IPVS: kubeproxyconfigv1alpha1.KubeProxyIPVSConfiguration{
							SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
						},
						Conntrack: kubeproxyconfigv1alpha1.KubeProxyConntrackConfiguration{
							Max:        utilpointer.Int32Ptr(2),
							MaxPerCore: utilpointer.Int32Ptr(1),
							Min:        utilpointer.Int32Ptr(1),
							TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
							TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
						},
					},
				},
			},
			msg: "must be IP:port",
		},
		{
			masterConfig: kubeadm.MasterConfiguration{
				KubeProxy: kubeadm.KubeProxy{
					Config: &kubeproxyconfigv1alpha1.KubeProxyConfiguration{
						BindAddress:        "10.10.12.11",
						HealthzBindAddress: "0.0.0.0:12345",
						// only MetricsBindAddress is invalid
						MetricsBindAddress: "127.0.0.1",
						ClusterCIDR:        "192.168.59.0/24",
						UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
						ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
						IPTables: kubeproxyconfigv1alpha1.KubeProxyIPTablesConfiguration{
							MasqueradeAll: true,
							SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
						},
						IPVS: kubeproxyconfigv1alpha1.KubeProxyIPVSConfiguration{
							SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
						},
						Conntrack: kubeproxyconfigv1alpha1.KubeProxyConntrackConfiguration{
							Max:        utilpointer.Int32Ptr(2),
							MaxPerCore: utilpointer.Int32Ptr(1),
							Min:        utilpointer.Int32Ptr(1),
							TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
							TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
						},
					},
				},
			},
			msg: "must be IP:port",
		},
		{
			masterConfig: kubeadm.MasterConfiguration{
				KubeProxy: kubeadm.KubeProxy{
					Config: &kubeproxyconfigv1alpha1.KubeProxyConfiguration{
						BindAddress:        "10.10.12.11",
						HealthzBindAddress: "0.0.0.0:12345",
						MetricsBindAddress: "127.0.0.1:10249",
						// only ClusterCIDR is invalid
						ClusterCIDR:      "192.168.59.0",
						UDPIdleTimeout:   metav1.Duration{Duration: 1 * time.Second},
						ConfigSyncPeriod: metav1.Duration{Duration: 1 * time.Second},
						IPTables: kubeproxyconfigv1alpha1.KubeProxyIPTablesConfiguration{
							MasqueradeAll: true,
							SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
						},
						IPVS: kubeproxyconfigv1alpha1.KubeProxyIPVSConfiguration{
							SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
						},
						Conntrack: kubeproxyconfigv1alpha1.KubeProxyConntrackConfiguration{
							Max:        utilpointer.Int32Ptr(2),
							MaxPerCore: utilpointer.Int32Ptr(1),
							Min:        utilpointer.Int32Ptr(1),
							TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
							TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
						},
					},
				},
			},
			msg: "must be a valid CIDR block (e.g. 10.100.0.0/16)",
		},
		{
			masterConfig: kubeadm.MasterConfiguration{
				KubeProxy: kubeadm.KubeProxy{
					Config: &kubeproxyconfigv1alpha1.KubeProxyConfiguration{
						BindAddress:        "10.10.12.11",
						HealthzBindAddress: "0.0.0.0:12345",
						MetricsBindAddress: "127.0.0.1:10249",
						ClusterCIDR:        "192.168.59.0/24",
						// only UDPIdleTimeout is invalid
						UDPIdleTimeout:   metav1.Duration{Duration: -1 * time.Second},
						ConfigSyncPeriod: metav1.Duration{Duration: 1 * time.Second},
						IPTables: kubeproxyconfigv1alpha1.KubeProxyIPTablesConfiguration{
							MasqueradeAll: true,
							SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
						},
						IPVS: kubeproxyconfigv1alpha1.KubeProxyIPVSConfiguration{
							SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
						},
						Conntrack: kubeproxyconfigv1alpha1.KubeProxyConntrackConfiguration{
							Max:        utilpointer.Int32Ptr(2),
							MaxPerCore: utilpointer.Int32Ptr(1),
							Min:        utilpointer.Int32Ptr(1),
							TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
							TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
						},
					},
				},
			},
			msg: "must be greater than 0",
		},
		{
			masterConfig: kubeadm.MasterConfiguration{
				KubeProxy: kubeadm.KubeProxy{
					Config: &kubeproxyconfigv1alpha1.KubeProxyConfiguration{
						BindAddress:        "10.10.12.11",
						HealthzBindAddress: "0.0.0.0:12345",
						MetricsBindAddress: "127.0.0.1:10249",
						ClusterCIDR:        "192.168.59.0/24",
						UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
						// only ConfigSyncPeriod is invalid
						ConfigSyncPeriod: metav1.Duration{Duration: -1 * time.Second},
						IPTables: kubeproxyconfigv1alpha1.KubeProxyIPTablesConfiguration{
							MasqueradeAll: true,
							SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
						},
						IPVS: kubeproxyconfigv1alpha1.KubeProxyIPVSConfiguration{
							SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
						},
						Conntrack: kubeproxyconfigv1alpha1.KubeProxyConntrackConfiguration{
							Max:        utilpointer.Int32Ptr(2),
							MaxPerCore: utilpointer.Int32Ptr(1),
							Min:        utilpointer.Int32Ptr(1),
							TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
							TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
						},
					},
				},
			},
			msg: "must be greater than 0",
		},
	}

	for i, errorCase := range errorCases {
		if errs := ValidateProxy(&errorCase.masterConfig, nil); len(errs) == 0 {
			t.Errorf("%d failed ValidateProxy: expected error for %s, but got no error", i, errorCase.msg)
		} else if !strings.Contains(errs[0].Error(), errorCase.msg) {
			t.Errorf("%d failed ValidateProxy: unexpected error: %v, expected: %s", i, errs[0], errorCase.msg)
		}
	}
}

func TestValidateArgSelection(t *testing.T) {
	var tests = []struct {
		name     string
		c        *kubeadm.NodeConfiguration
		expected bool
	}{
		{
			"invalid: DiscoveryToken and DiscoveryFile cannot both be set",
			&kubeadm.NodeConfiguration{
				DiscoveryFile:  "https://url/file.conf",
				DiscoveryToken: "abcdef.1234567890123456",
			},
			false,
		},
		{
			"invalid: DiscoveryToken or DiscoveryFile must be set",
			&kubeadm.NodeConfiguration{
				DiscoveryFile:  "",
				DiscoveryToken: "",
			},
			false,
		},
		{
			"invalid: DiscoveryTokenAPIServers not set",
			&kubeadm.NodeConfiguration{
				DiscoveryToken: "abcdef.1234567890123456",
			},
			false,
		},
		{
			"invalid: DiscoveryTokenCACertHashes cannot be used with DiscoveryFile",
			&kubeadm.NodeConfiguration{
				DiscoveryFile:              "https://url/file.conf",
				DiscoveryTokenCACertHashes: []string{"sha256:7173b809ca12ec5dee4506cd86be934c4596dd234ee82c0662eac04a8c2c71dc"},
			},
			false,
		},
		{
			"invalid: using token-based discovery without DiscoveryTokenCACertHashes and DiscoveryTokenUnsafeSkipCAVerification",
			&kubeadm.NodeConfiguration{
				DiscoveryToken:                         "abcdef.1234567890123456",
				DiscoveryTokenUnsafeSkipCAVerification: false,
				DiscoveryTokenAPIServers:               []string{"192.168.122.100:6443"},
			},
			false,
		},
		{
			"WARNING: kubeadm doesn't fully support multiple API Servers yet",
			&kubeadm.NodeConfiguration{
				DiscoveryToken:                         "abcdef.1234567890123456",
				DiscoveryTokenUnsafeSkipCAVerification: true,
				DiscoveryTokenAPIServers:               []string{"192.168.122.100:6443", "192.168.122.88:6443"},
			},
			true,
		},
		{
			"valid: DiscoveryFile with DiscoveryTokenAPIServers",
			&kubeadm.NodeConfiguration{
				DiscoveryFile:            "https://url/file.conf",
				DiscoveryTokenAPIServers: []string{"192.168.122.100:6443"},
			},
			true,
		},
		{
			"valid: DiscoveryFile without DiscoveryTokenAPIServers",
			&kubeadm.NodeConfiguration{
				DiscoveryFile: "https://url/file.conf",
			},
			true,
		},
		{
			"valid: using token-based discovery with DiscoveryTokenCACertHashes",
			&kubeadm.NodeConfiguration{
				DiscoveryToken:                         "abcdef.1234567890123456",
				DiscoveryTokenAPIServers:               []string{"192.168.122.100:6443"},
				DiscoveryTokenCACertHashes:             []string{"sha256:7173b809ca12ec5dee4506cd86be934c4596dd234ee82c0662eac04a8c2c71dc"},
				DiscoveryTokenUnsafeSkipCAVerification: false,
			},
			true,
		},
		{
			"valid: using token-based discovery with DiscoveryTokenCACertHashe but skip ca verification",
			&kubeadm.NodeConfiguration{
				DiscoveryToken:                         "abcdef.1234567890123456",
				DiscoveryTokenAPIServers:               []string{"192.168.122.100:6443"},
				DiscoveryTokenCACertHashes:             []string{"sha256:7173b809ca12ec5dee4506cd86be934c4596dd234ee82c0662eac04a8c2c71dc"},
				DiscoveryTokenUnsafeSkipCAVerification: true,
			},
			true,
		},
	}
	for _, rt := range tests {
		err := ValidateArgSelection(rt.c, nil).ToAggregate()
		if (err == nil) != rt.expected {
			t.Errorf(
				"%s test case failed: ValidateArgSelection:\n\texpected: %t\n\t  actual: %t",
				rt.name,
				rt.expected,
				(err == nil),
			)
		}
	}
}

func TestValidateJoinDiscoveryTokenAPIServer(t *testing.T) {
	var tests = []struct {
		s        *kubeadm.NodeConfiguration
		expected bool
	}{
		{
			&kubeadm.NodeConfiguration{
				DiscoveryTokenAPIServers: []string{"192.168.122.100"},
			},
			false,
		},
		{
			&kubeadm.NodeConfiguration{
				DiscoveryTokenAPIServers: []string{"192.168.122.100:6443"},
			},
			true,
		},
	}
	for _, rt := range tests {
		actual := ValidateJoinDiscoveryTokenAPIServer(rt.s, nil)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"failed ValidateJoinDiscoveryTokenAPIServer:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}

func TestValidateDiscoveryFile(t *testing.T) {
	tmpfile, err := ioutil.TempFile("/tmp", "test_discovery_file")
	if err != nil {
		t.Errorf("Error creating temporary file: %v", err)
	}
	defer os.Remove(tmpfile.Name())

	var tests = []struct {
		s        string
		expected bool
	}{
		{"foo", false},
		{"/foo/bar/file_which_i_believe_not_existing.conf", false},
		{tmpfile.Name(), true},
		{"http://[::1]a", false},
		{"http://url/file.conf", false},
		{"https://u r l/file.conf", false},
		{"https://url/file.conf", true},
	}
	for i, rt := range tests {
		actual := ValidateDiscoveryFile(rt.s, nil)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"%d: failed ValidateDiscoveryFile:\n\texpected: %t\n\t  actual: %t",
				i,
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}
