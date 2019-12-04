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

	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
)

func TestValidateToken(t *testing.T) {
	var tests = []struct {
		token    string
		expected bool
	}{
		{"772ef5.6b6baab1d4a0a171", true},
		{".6b6baab1d4a0a171", false},
		{"772ef5.", false},
		{"772ef5.6b6baab1d4a0a171", true},
		{".6b6baab1d4a0a171", false},
		{"772ef5.", false},
		{"abcdef.1234567890123456@foobar", false},
	}
	for _, rt := range tests {
		err := ValidateToken(rt.token, nil).ToAggregate()
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
		{"", "/some/path", true},                                                             // node name can't be empty
		{"INVALID-NODENAME", "/some/path", true},                                             // Upper cases is invalid
		{"invalid-nodename-", "/some/path", true},                                            // Can't have trailing dashes
		{"invalid-node?name", "/some/path", true},                                            // Unsupported characters
		{"valid-nodename", "/some/path", false},                                              // supported
		{"valid-nodename-with-numbers01234", "/some/path/with/numbers/01234/", false},        // supported, with numbers as well
		{"valid-nodename", kubeadmapiv1beta2.DefaultUrlScheme + "://" + "/some/path", false}, // supported, with socket url
		{"valid-nodename", "bla:///some/path", true},                                         // unsupported url scheme
		{"valid-nodename", ":::", true},                                                      // unparseable url
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
		{[]string{"*.my-hostname2", "*.my.other.subdomain"}, true},             // supported Wildcard DNS label
		{[]string{"**.my-hostname2", "my.other.subdomain"}, false},             // not a Wildcard DNS label
		{[]string{"*.*.my-hostname2", "my.other.subdomain"}, false},            // not a Wildcard DNS label
		{[]string{"a.*.my-hostname2", "my.other.subdomain"}, false},            // not a Wildcard DNS label
		{[]string{"*", "my.other.subdomain", "2001:db8::10"}, false},           // not a Wildcard DNS label
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
		name           string
		subnet         string
		minaddrs       int64
		checkDualStack bool
		expected       bool
	}{
		{"invalid missing CIDR", "", 0, false, false},
		{"invalid CIDR missing decimal points in IPv4 address and / mask", "1234", 0, false, false},
		{"invalid CIDR use of letters instead of numbers and / mask", "abc", 0, false, false},
		{"invalid IPv4 address provided instead of CIDR representation", "1.2.3.4", 0, false, false},
		{"invalid IPv6 address provided instead of CIDR representation", "2001:db8::1", 0, false, false},
		{"valid, but IPv4 CIDR too small. At least 10 addresses needed", "10.0.0.16/29", 10, false, false},
		{"valid, but IPv6 CIDR too small. At least 10 addresses needed", "2001:db8::/125", 10, false, false},
		{"valid IPv4 CIDR", "10.0.0.16/12", 10, false, true},
		{"valid IPv6 CIDR", "2001:db8::/98", 10, false, true},
		// dual-stack:
		{"invalid missing CIDR", "", 0, true, false},
		{"invalid only an IPv4 CIDR specified", "10.0.0.16/12", 10, true, false},
		{"invalid only an IPv6 CIDR specified", "2001:db8::/98", 10, true, false},
		{"invalid IPv4 address provided instead of CIDR representation", "1.2.3.4,2001:db8::/98", 0, true, false},
		{"invalid IPv6 address provided instead of CIDR representation", "2001:db8::1,10.0.0.16/12", 0, true, false},
		{"valid, but IPv4 CIDR too small. At least 10 addresses needed", "10.0.0.16/29,2001:db8::/98", 10, true, false},
		{"valid, but IPv6 CIDR too small. At least 10 addresses needed", "10.0.0.16/12,2001:db8::/125", 10, true, false},
		{"valid, but only IPv4 family addresses specified. IPv6 CIDR is necessary.", "10.0.0.16/12,192.168.0.0/16", 10, true, false},
		{"valid, but only IPv6 family addresses specified. IPv4 CIDR is necessary.", "2001:db8::/98,2005:db8::/98", 10, true, false},
		{"valid IPv4 and IPv6 CIDR", "10.0.0.16/12,2001:db8::/98", 10, true, true},
		{"valid IPv6 and IPv4 CIDR", "10.0.0.16/12,2001:db8::/98", 10, true, true},
	}
	for _, rt := range tests {
		actual := ValidateIPNetFromString(rt.subnet, rt.minaddrs, rt.checkDualStack, nil)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"%s test case failed :\n\texpected: %t\n\t  actual: %t\n\t  err(s): %v\n\t",
				rt.name,
				rt.expected,
				(len(actual) == 0),
				actual,
			)
		}
	}
}

func TestValidateHostPort(t *testing.T) {
	var tests = []struct {
		name     string
		s        string
		expected bool
	}{
		{
			name:     "Valid DNS address / port",
			s:        "cp.k8s.io:8081",
			expected: true,
		},
		{
			name:     "Valid DNS address",
			s:        "cp.k8s.io",
			expected: true,
		},
		{
			name:     "Valid IPv4 address / port",
			s:        "1.2.3.4:8081",
			expected: true,
		},
		{
			name:     "Valid IPv4 address",
			s:        "1.2.3.4",
			expected: true,
		},
		{
			name:     "Valid IPv6 address / port",
			s:        "[2001:db7::1]:8081",
			expected: true,
		},
		{
			name:     "Valid IPv6 address",
			s:        "2001:db7::1",
			expected: true,
		},
		{
			name:     "Invalid IPv4 address, but valid DNS",
			s:        "1.2.34",
			expected: true,
		},
		{
			name:     "Invalid DNS",
			s:        "a.B.c.d.e",
			expected: false,
		},
		{
			name:     "Invalid IPv6 address",
			s:        "2001:db7:1",
			expected: false,
		},
		{
			name:     "Invalid BindPort",
			s:        "1.2.3.4:0",
			expected: false,
		},
	}
	for _, rt := range tests {
		actual := ValidateHostPort(rt.s, nil)
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

func TestValidateAPIEndpoint(t *testing.T) {
	var tests = []struct {
		name     string
		s        *kubeadm.APIEndpoint
		expected bool
	}{
		{
			name: "Valid IPv4 address / port",
			s: &kubeadm.APIEndpoint{
				AdvertiseAddress: "4.5.6.7",
				BindPort:         6443,
			},
			expected: true,
		},
		{
			name: "Valid IPv6 address / port",
			s: &kubeadm.APIEndpoint{
				AdvertiseAddress: "2001:db7::2",
				BindPort:         6443,
			},
			expected: true,
		},
		{
			name: "Invalid IPv4 address",
			s: &kubeadm.APIEndpoint{
				AdvertiseAddress: "1.2.34",
				BindPort:         6443,
			},
			expected: false,
		},
		{
			name: "Invalid IPv6 address",
			s: &kubeadm.APIEndpoint{
				AdvertiseAddress: "2001:db7:1",
				BindPort:         6443,
			},
			expected: false,
		},
		{
			name: "Invalid BindPort",
			s: &kubeadm.APIEndpoint{
				AdvertiseAddress: "4.5.6.7",
				BindPort:         0,
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

//TODO: Create a separated test for ValidateClusterConfiguration
func TestValidateInitConfiguration(t *testing.T) {
	nodename := "valid-nodename"
	var tests = []struct {
		name     string
		s        *kubeadm.InitConfiguration
		expected bool
	}{
		{"invalid missing InitConfiguration",
			&kubeadm.InitConfiguration{}, false},
		{"invalid missing token with IPv4 service subnet",
			&kubeadm.InitConfiguration{
				LocalAPIEndpoint: kubeadm.APIEndpoint{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				ClusterConfiguration: kubeadm.ClusterConfiguration{
					Networking: kubeadm.Networking{
						ServiceSubnet: "10.96.0.1/12",
						DNSDomain:     "cluster.local",
					},
					CertificatesDir: "/some/cert/dir",
				},
				NodeRegistration: kubeadm.NodeRegistrationOptions{Name: nodename, CRISocket: "/some/path"},
			}, false},
		{"invalid missing token with IPv6 service subnet",
			&kubeadm.InitConfiguration{
				LocalAPIEndpoint: kubeadm.APIEndpoint{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				ClusterConfiguration: kubeadm.ClusterConfiguration{
					Networking: kubeadm.Networking{
						ServiceSubnet: "2001:db8::1/98",
						DNSDomain:     "cluster.local",
					},
					CertificatesDir: "/some/cert/dir",
				},
				NodeRegistration: kubeadm.NodeRegistrationOptions{Name: nodename, CRISocket: "/some/path"},
			}, false},
		{"invalid missing node name",
			&kubeadm.InitConfiguration{
				LocalAPIEndpoint: kubeadm.APIEndpoint{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				ClusterConfiguration: kubeadm.ClusterConfiguration{
					Networking: kubeadm.Networking{
						ServiceSubnet: "10.96.0.1/12",
						DNSDomain:     "cluster.local",
					},
					CertificatesDir: "/some/other/cert/dir",
				},
			}, false},
		{"valid InitConfiguration with incorrect IPv4 pod subnet",
			&kubeadm.InitConfiguration{
				LocalAPIEndpoint: kubeadm.APIEndpoint{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				ClusterConfiguration: kubeadm.ClusterConfiguration{
					Networking: kubeadm.Networking{
						ServiceSubnet: "10.96.0.1/12",
						DNSDomain:     "cluster.local",
						PodSubnet:     "10.0.1.15",
					},
					CertificatesDir: "/some/other/cert/dir",
				},
				NodeRegistration: kubeadm.NodeRegistrationOptions{Name: nodename, CRISocket: "/some/path"},
			}, false},
		{"valid InitConfiguration with IPv4 service subnet",
			&kubeadm.InitConfiguration{
				LocalAPIEndpoint: kubeadm.APIEndpoint{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				ClusterConfiguration: kubeadm.ClusterConfiguration{
					Etcd: kubeadm.Etcd{
						Local: &kubeadm.LocalEtcd{
							DataDir: "/some/path",
						},
					},
					Networking: kubeadm.Networking{
						ServiceSubnet: "10.96.0.1/12",
						DNSDomain:     "cluster.local",
						PodSubnet:     "10.0.1.15/16",
					},
					CertificatesDir: "/some/other/cert/dir",
				},
				NodeRegistration: kubeadm.NodeRegistrationOptions{Name: nodename, CRISocket: "/some/path"},
			}, true},
		{"valid InitConfiguration using IPv6 service subnet",
			&kubeadm.InitConfiguration{
				LocalAPIEndpoint: kubeadm.APIEndpoint{
					AdvertiseAddress: "1:2:3::4",
					BindPort:         3446,
				},
				ClusterConfiguration: kubeadm.ClusterConfiguration{
					Etcd: kubeadm.Etcd{
						Local: &kubeadm.LocalEtcd{
							DataDir: "/some/path",
						},
					},
					Networking: kubeadm.Networking{
						ServiceSubnet: "2001:db8::1/98",
						DNSDomain:     "cluster.local",
					},
					CertificatesDir: "/some/other/cert/dir",
				},
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
			CACertPath: "/some/cert.crt",
			Discovery: kubeadm.Discovery{
				BootstrapToken: &kubeadm.BootstrapTokenDiscovery{
					Token: "abcdef.1234567890123456@foobar",
				},
				File: &kubeadm.FileDiscovery{
					KubeConfigPath: "foo",
				},
			},
		}, false},
		{&kubeadm.JoinConfiguration{ // Pass without JoinControlPlane
			CACertPath: "/some/cert.crt",
			Discovery: kubeadm.Discovery{
				BootstrapToken: &kubeadm.BootstrapTokenDiscovery{
					Token:             "abcdef.1234567890123456",
					APIServerEndpoint: "1.2.3.4:6443",
					CACertHashes:      []string{"aaaa"},
				},
				TLSBootstrapToken: "abcdef.1234567890123456",
			},
			NodeRegistration: kubeadm.NodeRegistrationOptions{
				Name:      "aaa",
				CRISocket: "/var/run/dockershim.sock",
			},
		}, true},
		{&kubeadm.JoinConfiguration{ // Pass with JoinControlPlane
			CACertPath: "/some/cert.crt",
			Discovery: kubeadm.Discovery{
				BootstrapToken: &kubeadm.BootstrapTokenDiscovery{
					Token:             "abcdef.1234567890123456",
					APIServerEndpoint: "1.2.3.4:6443",
					CACertHashes:      []string{"aaaa"},
				},
				TLSBootstrapToken: "abcdef.1234567890123456",
			},
			NodeRegistration: kubeadm.NodeRegistrationOptions{
				Name:      "aaa",
				CRISocket: "/var/run/dockershim.sock",
			},
			ControlPlane: &kubeadm.JoinControlPlane{
				LocalAPIEndpoint: kubeadm.APIEndpoint{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         1234,
				},
			},
		}, true},
		{&kubeadm.JoinConfiguration{ // Fail JoinControlPlane.AdvertiseAddress validation
			CACertPath: "/some/cert.crt",
			Discovery: kubeadm.Discovery{
				BootstrapToken: &kubeadm.BootstrapTokenDiscovery{
					Token:             "abcdef.1234567890123456",
					APIServerEndpoint: "1.2.3.4:6443",
					CACertHashes:      []string{"aaaa"},
				},
				TLSBootstrapToken: "abcdef.1234567890123456",
			},
			NodeRegistration: kubeadm.NodeRegistrationOptions{
				Name:      "aaa",
				CRISocket: "/var/run/dockershim.sock",
			},
			ControlPlane: &kubeadm.JoinControlPlane{
				LocalAPIEndpoint: kubeadm.APIEndpoint{
					AdvertiseAddress: "aaa",
					BindPort:         1234,
				},
			},
		}, false},
		{&kubeadm.JoinConfiguration{ // Fail JoinControlPlane.BindPort validation
			CACertPath: "/some/cert.crt",
			Discovery: kubeadm.Discovery{
				BootstrapToken: &kubeadm.BootstrapTokenDiscovery{
					Token:             "abcdef.1234567890123456",
					APIServerEndpoint: "1.2.3.4:6443",
					CACertHashes:      []string{"aaaa"},
				},
				TLSBootstrapToken: "abcdef.1234567890123456",
			},
			NodeRegistration: kubeadm.NodeRegistrationOptions{
				Name:      "aaa",
				CRISocket: "/var/run/dockershim.sock",
			},
			ControlPlane: &kubeadm.JoinControlPlane{
				LocalAPIEndpoint: kubeadm.APIEndpoint{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         -1,
				},
			},
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
		{[]string{"--config=hello", "--ignore-preflight-errors=all"}, true},
		{[]string{"--config=hello", "--skip-token-print=true"}, true},
		{[]string{"--config=hello", "--ignore-preflight-errors=baz", "--skip-token-print"}, true},
		// Expected to fail, --config is mixed with the --foo flag
		{[]string{"--config=hello", "--ignore-preflight-errors=baz", "--foo=bar"}, false},
		{[]string{"--config=hello", "--foo=bar"}, false},
	}

	var cfgPath string
	var ignorePreflightErrors []string
	for _, rt := range tests {
		f := pflag.NewFlagSet("test", pflag.ContinueOnError)
		if f.Parsed() {
			t.Error("f.Parse() = true before Parse")
		}
		f.String("foo", "", "flag bound to config object")
		f.StringSliceVar(&ignorePreflightErrors, "ignore-preflight-errors", ignorePreflightErrors, "flag not bound to config object")
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
		{featureFlag{"Unknown": true}, false},
		{featureFlag{"Unknown": false}, false},
	}
	for _, rt := range tests {
		actual := ValidateFeatureGates(rt.featureGates, nil)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"failed featureGates %v:\n\texpected: %t\n\t  actual: %t",
				rt.featureGates,
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}

func TestValidateIgnorePreflightErrors(t *testing.T) {
	var tests = []struct {
		ignorePreflightErrorsFromCLI        []string
		ignorePreflightErrorsFromConfigFile []string
		expectedSet                         sets.String
		expectedError                       bool
	}{
		{ // empty lists in CLI and config file
			[]string{},
			[]string{},
			sets.NewString(),
			false,
		},
		{ // empty list in CLI only
			[]string{},
			[]string{"a"},
			sets.NewString("a"),
			false,
		},
		{ // empty list in config file only
			[]string{"a"},
			[]string{},
			sets.NewString("a"),
			false,
		},
		{ // no duplicates, no overlap
			[]string{"a", "b"},
			[]string{"c", "d"},
			sets.NewString("a", "b", "c", "d"),
			false,
		},
		{ // some duplicates, with some overlapping duplicates
			[]string{"a", "b", "a"},
			[]string{"c", "b"},
			sets.NewString("a", "b", "c"),
			false,
		},
		{ // non-duplicate, but 'all' present together with individual checks in CLI
			[]string{"a", "b", "all"},
			[]string{},
			sets.NewString(),
			true,
		},
		{ // empty list in CLI, but 'all' present in config file, which is forbidden
			[]string{},
			[]string{"all"},
			sets.NewString(),
			true,
		},
		{ // non-duplicate, but 'all' present in config file, which is forbidden
			[]string{"a", "b"},
			[]string{"all"},
			sets.NewString(),
			true,
		},
		{ // non-duplicate, but 'all' present in CLI, while values are in config file, which is forbidden
			[]string{"all"},
			[]string{"a", "b"},
			sets.NewString(),
			true,
		},
		{ // skip all checks
			[]string{"all"},
			[]string{},
			sets.NewString("all"),
			false,
		},
	}
	for _, rt := range tests {
		result, err := ValidateIgnorePreflightErrors(rt.ignorePreflightErrorsFromCLI, rt.ignorePreflightErrorsFromConfigFile)
		switch {
		case err != nil && !rt.expectedError:
			t.Errorf("ValidateIgnorePreflightErrors: unexpected error for input (%s, %s), error: %v", rt.ignorePreflightErrorsFromCLI, rt.ignorePreflightErrorsFromConfigFile, err)
		case err == nil && rt.expectedError:
			t.Errorf("ValidateIgnorePreflightErrors: expected error for input (%s, %s) but got: %v", rt.ignorePreflightErrorsFromCLI, rt.ignorePreflightErrorsFromConfigFile, result)
		case err == nil && !result.Equal(rt.expectedSet):
			t.Errorf("ValidateIgnorePreflightErrors: expected (%v) for input (%s, %s) but got: %v", rt.expectedSet, rt.ignorePreflightErrorsFromCLI, rt.ignorePreflightErrorsFromConfigFile, result)
		}
	}
}

func TestValidateDiscovery(t *testing.T) {
	var tests = []struct {
		name     string
		d        *kubeadm.Discovery
		expected bool
	}{
		{
			"invalid: .BootstrapToken and .File cannot both be set",
			&kubeadm.Discovery{
				BootstrapToken: &kubeadm.BootstrapTokenDiscovery{
					Token: "abcdef.1234567890123456",
				},
				File: &kubeadm.FileDiscovery{
					KubeConfigPath: "https://url/file.conf",
				},
			},
			false,
		},
		{
			"invalid: .BootstrapToken or .File must be set",
			&kubeadm.Discovery{},
			false,
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			err := ValidateDiscovery(rt.d, nil).ToAggregate()
			if (err == nil) != rt.expected {
				t.Errorf(
					"test case failed: ValidateDiscovery:\n\texpected: %t\n\t  actual: %t",
					rt.expected,
					(err == nil),
				)
			}
		})
	}
}

func TestValidateDiscoveryBootstrapToken(t *testing.T) {
	var tests = []struct {
		name     string
		btd      *kubeadm.BootstrapTokenDiscovery
		expected bool
	}{
		{
			"invalid: .APIServerEndpoint not set",
			&kubeadm.BootstrapTokenDiscovery{
				Token: "abcdef.1234567890123456",
			},
			false,
		},
		{
			"invalid: using token-based discovery without .BootstrapToken.CACertHashes and .BootstrapToken.UnsafeSkipCAVerification",
			&kubeadm.BootstrapTokenDiscovery{
				Token:                    "abcdef.1234567890123456",
				APIServerEndpoint:        "192.168.122.100:6443",
				UnsafeSkipCAVerification: false,
			},
			false,
		},
		{
			"valid: using token-based discovery with .BootstrapToken.CACertHashes",
			&kubeadm.BootstrapTokenDiscovery{
				Token:                    "abcdef.1234567890123456",
				APIServerEndpoint:        "192.168.122.100:6443",
				CACertHashes:             []string{"sha256:7173b809ca12ec5dee4506cd86be934c4596dd234ee82c0662eac04a8c2c71dc"},
				UnsafeSkipCAVerification: false,
			},
			true,
		},
		{
			"valid: using token-based discovery with .BootstrapToken.CACertHashe but skip ca verification",
			&kubeadm.BootstrapTokenDiscovery{
				Token:                    "abcdef.1234567890123456",
				APIServerEndpoint:        "192.168.122.100:6443",
				CACertHashes:             []string{"sha256:7173b809ca12ec5dee4506cd86be934c4596dd234ee82c0662eac04a8c2c71dc"},
				UnsafeSkipCAVerification: true,
			},
			true,
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			err := ValidateDiscoveryBootstrapToken(rt.btd, nil).ToAggregate()
			if (err == nil) != rt.expected {
				t.Errorf(
					"test case failed: ValidateDiscoveryBootstrapToken:\n\texpected: %t\n\t  actual: %t",
					rt.expected,
					(err == nil),
				)
			}
		})
	}
}

func TestValidateDiscoveryTokenAPIServer(t *testing.T) {
	var tests = []struct {
		apiServerEndpoint string
		expected          bool
	}{
		{
			"192.168.122.100",
			false,
		},
		{
			"192.168.122.100:6443",
			true,
		},
	}
	for _, rt := range tests {
		actual := ValidateDiscoveryTokenAPIServer(rt.apiServerEndpoint, nil)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"failed ValidateDiscoveryTokenAPIServer:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}

func TestValidateDiscoveryKubeConfigPath(t *testing.T) {
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
		actual := ValidateDiscoveryKubeConfigPath(rt.s, nil)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"%d: failed ValidateDiscoveryKubeConfigPath:\n\texpected: %t\n\t  actual: %t",
				i,
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}
