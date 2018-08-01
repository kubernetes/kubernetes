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
	"testing"
	"time"

	"github.com/spf13/pflag"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
	"k8s.io/kubernetes/pkg/proxy/apis/kubeproxyconfig"
	utilpointer "k8s.io/utils/pointer"
)

func TestValidateToken(t *testing.T) {
	var tests = []struct {
		c        *kubeadm.JoinConfiguration
		f        *field.Path
		expected bool
	}{
		{&kubeadm.JoinConfiguration{Token: "772ef5.6b6baab1d4a0a171", DiscoveryTokenAPIServers: []string{"192.168.122.100:6443"}}, nil, true},
		{&kubeadm.JoinConfiguration{Token: ".6b6baab1d4a0a171", DiscoveryTokenAPIServers: []string{"192.168.122.100:6443"}}, nil, false},
		{&kubeadm.JoinConfiguration{Token: "772ef5.", DiscoveryTokenAPIServers: []string{"192.168.122.100:6443"}}, nil, false},
		{&kubeadm.JoinConfiguration{Token: "772ef5.6b6baab1d4a0a171", DiscoveryTokenAPIServers: []string{"2001:db8::100:6443"}}, nil, true},
		{&kubeadm.JoinConfiguration{Token: ".6b6baab1d4a0a171", DiscoveryTokenAPIServers: []string{"2001:db8::100:6443"}}, nil, false},
		{&kubeadm.JoinConfiguration{Token: "772ef5.", DiscoveryTokenAPIServers: []string{"2001:db8::100:6443"}}, nil, false},
		{&kubeadm.JoinConfiguration{Token: "abcdef.1234567890123456@foobar", DiscoveryTokenAPIServers: []string{"192.168.122.100:6443"}}, nil, false},
	}
	for _, rt := range tests {
		err := ValidateToken(rt.c.Token, rt.f).ToAggregate()
		if (err == nil) != rt.expected {
			t.Errorf(
				"failed ValidateToken:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(err == nil),
			)
		}
	}
}

func TestValidateValidateTokenUsages(t *testing.T) {
	var tests = []struct {
		u        []string
		f        *field.Path
		expected bool
	}{
		{[]string{}, nil, true},                            // supported (no usages)
		{[]string{"signing", "authentication"}, nil, true}, // supported
		{[]string{"something else"}, nil, false},           // usage not supported
	}
	for _, rt := range tests {
		actual := ValidateTokenUsages(rt.u, rt.f)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"failed ValidateTokenUsages:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}

func TestValidateTokenGroups(t *testing.T) {
	var tests = []struct {
		u        []string
		g        []string
		f        *field.Path
		expected bool
	}{
		{[]string{"some usage"}, []string{"some group"}, nil, false},                       // groups doesn't makes sense if usage authentication
		{[]string{"authentication"}, []string{"some group"}, nil, false},                   // group not supported
		{[]string{"authentication"}, []string{"system:bootstrappers:anygroup"}, nil, true}, // supported
	}
	for _, rt := range tests {
		actual := ValidateTokenGroups(rt.u, rt.g, rt.f)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"failed ValidateTokenGroups:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}

func TestValidateNodeRegistrationOptions(t *testing.T) {
	var tests = []struct {
		nodeName       string
		criSocket      string
		expectedErrors bool
	}{
		{"", "/some/path", true},                                                              // node name can't be empty
		{"INVALID-NODENAME", "/some/path", true},                                              // Upper cases is invalid
		{"invalid-nodename-", "/some/path", true},                                             // Can't have trailing dashes
		{"invalid-node?name", "/some/path", true},                                             // Unsupported characters
		{"valid-nodename", "/some/path", false},                                               // supported
		{"valid-nodename-with-numbers01234", "/some/path/with/numbers/01234/", false},         // supported, with numbers as well
		{"valid-nodename", kubeadmapiv1alpha3.DefaultUrlScheme + "://" + "/some/path", false}, // supported, with socket url
		{"valid-nodename", "bla:///some/path", true},                                          // unsupported url scheme
		{"valid-nodename", ":::", true},                                                       // unparseable url
	}
	for _, rt := range tests {
		nro := kubeadm.NodeRegistrationOptions{Name: rt.nodeName, CRISocket: rt.criSocket}
		actual := ValidateNodeRegistrationOptions(&nro, field.NewPath("nodeRegistration"))
		actualErrors := len(actual) > 0
		if actualErrors != rt.expectedErrors {
			t.Errorf(
				"failed ValidateNodeRegistrationOptions: value: %v\n\texpected: %t\n\t  actual: %t",
				nro,
				rt.expectedErrors,
				actualErrors,
			)
		}
	}
}

func TestValidateCertSANs(t *testing.T) {
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
		actual := ValidateCertSANs(rt.sans, nil)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"failed ValidateCertSANs:\n\texpected: %t\n\t  actual: %t",
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
		s        *kubeadm.InitConfiguration
		expected bool
	}{
		{
			name:     "Missing configuration",
			s:        &kubeadm.InitConfiguration{},
			expected: false,
		},
		{
			name: "Valid DNS ControlPlaneEndpoint (with port), AdvertiseAddress and default port",
			s: &kubeadm.InitConfiguration{
				API: kubeadm.API{
					ControlPlaneEndpoint: "cp.k8s.io:8081",
					AdvertiseAddress:     "4.5.6.7",
					BindPort:             6443,
				},
			},
			expected: true,
		},
		{
			name: "Valid IPv4 ControlPlaneEndpoint (with port), AdvertiseAddress and default port",
			s: &kubeadm.InitConfiguration{
				API: kubeadm.API{
					ControlPlaneEndpoint: "1.2.3.4:8081",
					AdvertiseAddress:     "4.5.6.7",
					BindPort:             6443,
				},
			},
			expected: true,
		},
		{
			name: "Valid IPv6 ControlPlaneEndpoint (with port), ControlPlaneEndpoint and port",
			s: &kubeadm.InitConfiguration{
				API: kubeadm.API{
					ControlPlaneEndpoint: "[2001:db7::1]:8081",
					AdvertiseAddress:     "2001:db7::2",
					BindPort:             6443,
				},
			},
			expected: true,
		},
		{
			name: "Valid DNS ControlPlaneEndpoint (without port), AdvertiseAddress and default port",
			s: &kubeadm.InitConfiguration{
				API: kubeadm.API{
					ControlPlaneEndpoint: "cp.k8s.io",
					AdvertiseAddress:     "4.5.6.7",
					BindPort:             6443,
				},
			},
			expected: true,
		},
		{
			name: "Valid IPv4 ControlPlaneEndpoint (without port), AdvertiseAddress and default port",
			s: &kubeadm.InitConfiguration{
				API: kubeadm.API{
					ControlPlaneEndpoint: "1.2.3.4",
					AdvertiseAddress:     "4.5.6.7",
					BindPort:             6443,
				},
			},
			expected: true,
		},
		{
			name: "Valid IPv6 ControlPlaneEndpoint (without port), ControlPlaneEndpoint and port",
			s: &kubeadm.InitConfiguration{
				API: kubeadm.API{
					ControlPlaneEndpoint: "2001:db7::1",
					AdvertiseAddress:     "2001:db7::2",
					BindPort:             6443,
				},
			},
			expected: true,
		},
		{
			name: "Valid IPv4 AdvertiseAddress and default port",
			s: &kubeadm.InitConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
			},
			expected: true,
		},
		{
			name: "Valid IPv6 AdvertiseAddress and port",
			s: &kubeadm.InitConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "2001:db7::1",
					BindPort:         3446,
				},
			},
			expected: true,
		},
		{
			name: "Invalid IPv4 AdvertiseAddress",
			s: &kubeadm.InitConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "1.2.34",
					BindPort:         6443,
				},
			},
			expected: false,
		},
		{
			name: "Invalid IPv6 AdvertiseAddress",
			s: &kubeadm.InitConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "2001:db7:1",
					BindPort:         3446,
				},
			},
			expected: false,
		},
		{
			name: "Invalid BindPort",
			s: &kubeadm.InitConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         0,
				},
			},
			expected: false,
		},
		{
			name: "Invalid DNS ControlPlaneEndpoint",
			s: &kubeadm.InitConfiguration{
				API: kubeadm.API{
					ControlPlaneEndpoint: "bad!!.k8s.io",
				},
			},
			expected: false,
		},
		{
			name: "Invalid ipv4 ControlPlaneEndpoint",
			s: &kubeadm.InitConfiguration{
				API: kubeadm.API{
					ControlPlaneEndpoint: "1..3.4",
				},
			},
			expected: false,
		},
		{
			name: "Invalid ipv6 ControlPlaneEndpoint",
			s: &kubeadm.InitConfiguration{
				API: kubeadm.API{
					ControlPlaneEndpoint: "1200::AB00:1234::2552:7777:1313",
				},
			},
			expected: false,
		},
		{
			name: "Invalid ControlPlaneEndpoint port",
			s: &kubeadm.InitConfiguration{
				API: kubeadm.API{
					ControlPlaneEndpoint: "1.2.3.4:0",
				},
			},
			expected: false,
		},
	}
	for _, rt := range tests {
		actual := ValidateAPIEndpoint(&rt.s.API, nil)
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

func TestValidateInitConfiguration(t *testing.T) {
	nodename := "valid-nodename"
	var tests = []struct {
		name     string
		s        *kubeadm.InitConfiguration
		expected bool
	}{
		{"invalid missing master configuration",
			&kubeadm.InitConfiguration{}, false},
		{"invalid missing token with IPv4 service subnet",
			&kubeadm.InitConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				Networking: kubeadm.Networking{
					ServiceSubnet: "10.96.0.1/12",
					DNSDomain:     "cluster.local",
				},
				CertificatesDir:  "/some/cert/dir",
				NodeRegistration: kubeadm.NodeRegistrationOptions{Name: nodename, CRISocket: "/some/path"},
			}, false},
		{"invalid missing token with IPv6 service subnet",
			&kubeadm.InitConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				Networking: kubeadm.Networking{
					ServiceSubnet: "2001:db8::1/98",
					DNSDomain:     "cluster.local",
				},
				CertificatesDir:  "/some/cert/dir",
				NodeRegistration: kubeadm.NodeRegistrationOptions{Name: nodename, CRISocket: "/some/path"},
			}, false},
		{"invalid missing node name",
			&kubeadm.InitConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				Networking: kubeadm.Networking{
					ServiceSubnet: "10.96.0.1/12",
					DNSDomain:     "cluster.local",
				},
				CertificatesDir: "/some/other/cert/dir",
			}, false},
		{"valid master configuration with incorrect IPv4 pod subnet",
			&kubeadm.InitConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				Networking: kubeadm.Networking{
					ServiceSubnet: "10.96.0.1/12",
					DNSDomain:     "cluster.local",
					PodSubnet:     "10.0.1.15",
				},
				CertificatesDir:  "/some/other/cert/dir",
				NodeRegistration: kubeadm.NodeRegistrationOptions{Name: nodename, CRISocket: "/some/path"},
			}, false},
		{"valid master configuration with IPv4 service subnet",
			&kubeadm.InitConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				Etcd: kubeadm.Etcd{
					Local: &kubeadm.LocalEtcd{
						DataDir: "/some/path",
					},
				},
				ComponentConfigs: kubeadm.ComponentConfigs{
					KubeProxy: &kubeproxyconfig.KubeProxyConfiguration{
						BindAddress:        "192.168.59.103",
						HealthzBindAddress: "0.0.0.0:10256",
						MetricsBindAddress: "127.0.0.1:10249",
						ClusterCIDR:        "192.168.59.0/24",
						UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
						ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
						IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
							MasqueradeAll: true,
							SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
						},
						IPVS: kubeproxyconfig.KubeProxyIPVSConfiguration{
							SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
						},
						Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
							Max:        utilpointer.Int32Ptr(2),
							MaxPerCore: utilpointer.Int32Ptr(1),
							Min:        utilpointer.Int32Ptr(1),
							TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
							TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
						},
					},
				},
				Networking: kubeadm.Networking{
					ServiceSubnet: "10.96.0.1/12",
					DNSDomain:     "cluster.local",
					PodSubnet:     "10.0.1.15/16",
				},
				CertificatesDir:  "/some/other/cert/dir",
				NodeRegistration: kubeadm.NodeRegistrationOptions{Name: nodename, CRISocket: "/some/path"},
			}, true},
		{"valid master configuration using IPv6 service subnet",
			&kubeadm.InitConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "1:2:3::4",
					BindPort:         3446,
				},
				Etcd: kubeadm.Etcd{
					Local: &kubeadm.LocalEtcd{
						DataDir: "/some/path",
					},
				},
				ComponentConfigs: kubeadm.ComponentConfigs{
					KubeProxy: &kubeproxyconfig.KubeProxyConfiguration{
						BindAddress:        "192.168.59.103",
						HealthzBindAddress: "0.0.0.0:10256",
						MetricsBindAddress: "127.0.0.1:10249",
						ClusterCIDR:        "192.168.59.0/24",
						UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
						ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
						IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
							MasqueradeAll: true,
							SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
						},
						IPVS: kubeproxyconfig.KubeProxyIPVSConfiguration{
							SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
							MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
						},
						Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
							Max:        utilpointer.Int32Ptr(2),
							MaxPerCore: utilpointer.Int32Ptr(1),
							Min:        utilpointer.Int32Ptr(1),
							TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
							TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
						},
					},
				},
				Networking: kubeadm.Networking{
					ServiceSubnet: "2001:db8::1/98",
					DNSDomain:     "cluster.local",
				},
				CertificatesDir:  "/some/other/cert/dir",
				NodeRegistration: kubeadm.NodeRegistrationOptions{Name: nodename, CRISocket: "/some/path"},
			}, true},
	}
	for _, rt := range tests {
		actual := ValidateInitConfiguration(rt.s)
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

func TestValidateJoinConfiguration(t *testing.T) {
	var tests = []struct {
		s        *kubeadm.JoinConfiguration
		expected bool
	}{
		{&kubeadm.JoinConfiguration{}, false},
		{&kubeadm.JoinConfiguration{
			DiscoveryFile:  "foo",
			DiscoveryToken: "abcdef.1234567890123456@foobar",
			CACertPath:     "/some/cert.crt",
		}, false},
	}
	for _, rt := range tests {
		actual := ValidateJoinConfiguration(rt.s)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"failed ValidateJoinConfiguration:\n\texpected: %t\n\t  actual: %t",
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

func TestValidateArgSelection(t *testing.T) {
	var tests = []struct {
		name     string
		c        *kubeadm.JoinConfiguration
		expected bool
	}{
		{
			"invalid: DiscoveryToken and DiscoveryFile cannot both be set",
			&kubeadm.JoinConfiguration{
				DiscoveryFile:  "https://url/file.conf",
				DiscoveryToken: "abcdef.1234567890123456",
			},
			false,
		},
		{
			"invalid: DiscoveryToken or DiscoveryFile must be set",
			&kubeadm.JoinConfiguration{
				DiscoveryFile:  "",
				DiscoveryToken: "",
			},
			false,
		},
		{
			"invalid: DiscoveryTokenAPIServers not set",
			&kubeadm.JoinConfiguration{
				DiscoveryToken: "abcdef.1234567890123456",
			},
			false,
		},
		{
			"invalid: DiscoveryTokenCACertHashes cannot be used with DiscoveryFile",
			&kubeadm.JoinConfiguration{
				DiscoveryFile:              "https://url/file.conf",
				DiscoveryTokenCACertHashes: []string{"sha256:7173b809ca12ec5dee4506cd86be934c4596dd234ee82c0662eac04a8c2c71dc"},
			},
			false,
		},
		{
			"invalid: using token-based discovery without DiscoveryTokenCACertHashes and DiscoveryTokenUnsafeSkipCAVerification",
			&kubeadm.JoinConfiguration{
				DiscoveryToken:                         "abcdef.1234567890123456",
				DiscoveryTokenUnsafeSkipCAVerification: false,
				DiscoveryTokenAPIServers:               []string{"192.168.122.100:6443"},
			},
			false,
		},
		{
			"WARNING: kubeadm doesn't fully support multiple API Servers yet",
			&kubeadm.JoinConfiguration{
				DiscoveryToken:                         "abcdef.1234567890123456",
				DiscoveryTokenUnsafeSkipCAVerification: true,
				DiscoveryTokenAPIServers:               []string{"192.168.122.100:6443", "192.168.122.88:6443"},
			},
			true,
		},
		{
			"valid: DiscoveryFile with DiscoveryTokenAPIServers",
			&kubeadm.JoinConfiguration{
				DiscoveryFile:            "https://url/file.conf",
				DiscoveryTokenAPIServers: []string{"192.168.122.100:6443"},
			},
			true,
		},
		{
			"valid: DiscoveryFile without DiscoveryTokenAPIServers",
			&kubeadm.JoinConfiguration{
				DiscoveryFile: "https://url/file.conf",
			},
			true,
		},
		{
			"valid: using token-based discovery with DiscoveryTokenCACertHashes",
			&kubeadm.JoinConfiguration{
				DiscoveryToken:                         "abcdef.1234567890123456",
				DiscoveryTokenAPIServers:               []string{"192.168.122.100:6443"},
				DiscoveryTokenCACertHashes:             []string{"sha256:7173b809ca12ec5dee4506cd86be934c4596dd234ee82c0662eac04a8c2c71dc"},
				DiscoveryTokenUnsafeSkipCAVerification: false,
			},
			true,
		},
		{
			"valid: using token-based discovery with DiscoveryTokenCACertHashe but skip ca verification",
			&kubeadm.JoinConfiguration{
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
		s        *kubeadm.JoinConfiguration
		expected bool
	}{
		{
			&kubeadm.JoinConfiguration{
				DiscoveryTokenAPIServers: []string{"192.168.122.100"},
			},
			false,
		},
		{
			&kubeadm.JoinConfiguration{
				DiscoveryTokenAPIServers: []string{"192.168.122.100:6443"},
			},
			true,
		},
	}
	for _, rt := range tests {
		actual := ValidateJoinDiscoveryTokenAPIServer(rt.s.DiscoveryTokenAPIServers, nil)
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
