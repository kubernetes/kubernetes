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
	"testing"

	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestValidateTokenDiscovery(t *testing.T) {
	var tests = []struct {
		c        *kubeadm.NodeConfiguration
		f        *field.Path
		expected bool
	}{
		{&kubeadm.NodeConfiguration{Token: "772ef5.6b6baab1d4a0a171", DiscoveryTokenAPIServers: []string{"192.168.122.100:9898"}}, nil, true},
		{&kubeadm.NodeConfiguration{Token: ".6b6baab1d4a0a171", DiscoveryTokenAPIServers: []string{"192.168.122.100:9898"}}, nil, false},
		{&kubeadm.NodeConfiguration{Token: "772ef5.", DiscoveryTokenAPIServers: []string{"192.168.122.100:9898"}}, nil, false},
		{&kubeadm.NodeConfiguration{Token: "772ef5.6b6baab1d4a0a171", DiscoveryTokenAPIServers: []string{"2001:db8::100:9898"}}, nil, true},
		{&kubeadm.NodeConfiguration{Token: ".6b6baab1d4a0a171", DiscoveryTokenAPIServers: []string{"2001:db8::100:9898"}}, nil, false},
		{&kubeadm.NodeConfiguration{Token: "772ef5.", DiscoveryTokenAPIServers: []string{"2001:db8::100:9898"}}, nil, false},
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
		{"valid master configuration with IPv4 service subnet",
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
				NodeName:        nodename,
			}, true},
		{"valid master configuration using IPv6 service subnet",
			&kubeadm.MasterConfiguration{
				API: kubeadm.API{
					AdvertiseAddress: "1:2:3::4",
					BindPort:         3446,
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
