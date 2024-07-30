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
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/spf13/pflag"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
)

func TestValidateToken(t *testing.T) {
	var tests = []struct {
		token    string
		expected bool
	}{
		{"772ef5.6b6baab1d4a0a171", true},
		{".6b6baab1d4a0a171", false},
		{"772ef5.", false},
		{"abcdef.1234567890123456@foobar", false},
	}
	for _, rt := range tests {
		err := ValidateToken(rt.token, nil).ToAggregate()
		if (err == nil) != rt.expected {
			t.Errorf(
				"failed ValidateToken:\n\ttoken: %q\n\t  expected: %t, got: %t",
				rt.token,
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
		expectedErrors bool
	}{
		{"", true},                  // node name can't be empty
		{"INVALID-NODENAME", true},  // Upper cases is invalid
		{"invalid-nodename-", true}, // Can't have trailing dashes
		{"invalid-node?name", true}, // Unsupported characters
		{"valid-nodename", false},   // supported
		// test cases for criSocket are covered in TestValidateSocketPath
	}
	criPath := fmt.Sprintf("%s:///some/path", kubeadmapiv1.DefaultContainerRuntimeURLScheme)
	for _, rt := range tests {
		nro := kubeadmapi.NodeRegistrationOptions{Name: rt.nodeName, CRISocket: criPath}
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

func TestValidatePort(t *testing.T) {
	var tests = []struct {
		name        string
		port        int32
		expectedErr bool
	}{
		{"negative number port", -1234, true},
		{"zero number port", 0, true},
		{"minimum valid value port", 1, false},
		{"valid value port", 300, false},
		{"maximum valid value port", 65535, false},
		{"if port greater than 65535", 65538, true},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			allErrs := ValidatePort(rt.port, nil)
			if len(allErrs) > 0 {
				find := strings.Contains(allErrs[0].Error(), "port number is not valid")
				if find != rt.expectedErr {
					t.Errorf(
						"test case failed :\n\t   err(s): %v\n\t", allErrs[0].Error(),
					)
				}
			}
		})
	}
}

func TestValidateIPNetFromString(t *testing.T) {
	var tests = []struct {
		name     string
		subnet   string
		minaddrs int64
		expected bool
	}{
		// dual-stack:
		{"invalid missing CIDR", "", 0, false},
		{"valid dual-stack enabled but only an IPv4 CIDR specified", "10.0.0.16/12", 10, true},
		{"valid dual-stack enabled but only an IPv6 CIDR specified", "2001:db8::/98", 10, true},
		{"invalid IPv4 address provided instead of CIDR representation", "1.2.3.4,2001:db8::/98", 0, false},
		{"invalid IPv6 address provided instead of CIDR representation", "2001:db8::1,10.0.0.16/12", 0, false},
		{"valid, but IPv4 CIDR too small. At least 10 addresses needed", "10.0.0.16/29,2001:db8::/98", 10, false},
		{"valid, but IPv6 CIDR too small. At least 10 addresses needed", "10.0.0.16/12,2001:db8::/125", 10, false},
		{"valid, but only IPv4 family addresses specified. IPv6 CIDR is necessary.", "10.0.0.16/12,192.168.0.0/16", 10, false},
		{"valid, but only IPv6 family addresses specified. IPv4 CIDR is necessary.", "2001:db8::/98,2005:db8::/98", 10, false},
		{"valid IPv4 and IPv6 CIDR", "10.0.0.16/12,2001:db8::/98", 10, true},
		{"valid IPv6 and IPv4 CIDR", "10.0.0.16/12,2001:db8::/98", 10, true},
		{"invalid IPv6 and IPv4 CIDR with more than 2 subnets", "10.0.0.16/12,2001:db8::/98,192.168.0.0/16", 10, false},
		{"invalid IPv6 and IPv4 CIDR with more than 2 subnets", "10.0.0.16/12,2001:db8::/98,192.168.0.0/16,a.b.c.d/24", 10, false},
	}
	for _, rt := range tests {
		actual := ValidateIPNetFromString(rt.subnet, rt.minaddrs, nil)
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

func TestValidatePodSubnetNodeMask(t *testing.T) {
	var tests = []struct {
		name        string
		subnet      string
		cmExtraArgs []kubeadmapi.Arg
		expected    bool
	}{
		// dual-stack:
		{"dual IPv4 only, but mask too small. Default node-mask", "10.0.0.16/29", nil, false},
		{"dual IPv4 only, but mask too small. Configured node-mask", "10.0.0.16/24", []kubeadmapi.Arg{{Name: "node-cidr-mask-size-ipv4", Value: "23"}}, false},
		{"dual IPv6 only, but mask too small. Default node-mask", "2001:db8::1/112", nil, false},
		{"dual IPv6 only, but mask too small. Configured node-mask", "2001:db8::1/64", []kubeadmapi.Arg{{Name: "node-cidr-mask-size-ipv6", Value: "24"}}, false},
		{"dual IPv6 only, but mask difference greater than 16. Default node-mask", "2001:db8::1/12", nil, false},
		{"dual IPv6 only, but mask difference greater than 16. Configured node-mask", "2001:db8::1/64", []kubeadmapi.Arg{{Name: "node-cidr-mask-size-ipv6", Value: "120"}}, false},
		{"dual IPv4 only CIDR", "10.0.0.16/12", nil, true},
		{"dual IPv6 only CIDR", "2001:db8::/48", nil, true},
		{"dual, but IPv4 mask too small. Default node-mask", "10.0.0.16/29,2001:db8::/48", nil, false},
		{"dual, but IPv4 mask too small. Configured node-mask", "10.0.0.16/24,2001:db8::/48", []kubeadmapi.Arg{{Name: "node-cidr-mask-size-ipv4", Value: "23"}}, false},
		{"dual, but IPv6 mask too small. Default node-mask", "2001:db8::1/112,10.0.0.16/16", nil, false},
		{"dual, but IPv6 mask too small. Configured node-mask", "10.0.0.16/16,2001:db8::1/64", []kubeadmapi.Arg{{Name: "node-cidr-mask-size-ipv6", Value: "24"}}, false},
		{"dual, but mask difference greater than 16. Default node-mask", "2001:db8::1/12,10.0.0.16/16", nil, false},
		{"dual, but mask difference greater than 16. Configured node-mask", "10.0.0.16/16,2001:db8::1/64", []kubeadmapi.Arg{{Name: "node-cidr-mask-size-ipv6", Value: "120"}}, false},
		{"dual IPv4 IPv6", "2001:db8::/48,10.0.0.16/12", nil, true},
		{"dual IPv6 IPv4", "2001:db8::/48,10.0.0.16/12", nil, true},
	}
	for _, rt := range tests {
		cfg := &kubeadmapi.ClusterConfiguration{
			ControllerManager: kubeadmapi.ControlPlaneComponent{
				ExtraArgs: rt.cmExtraArgs,
			},
		}
		actual := ValidatePodSubnetNodeMask(rt.subnet, cfg, nil)
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

func TestValidateServiceSubnetSize(t *testing.T) {
	var tests = []struct {
		name     string
		subnet   string
		expected bool
	}{
		{"single IPv4, but mask too large.", "10.0.0.16/2", false},
		{"single IPv6, but mask too large.", "2001:db8::1/64", false},
		{"single IPv4 CIDR", "10.0.0.16/12", true},
		{"single IPv6 CIDR", "2001:db8::/112", true},
		// dual-stack:
		{"dual, but IPv4 mask too large.", "2001:db8::1/112,10.0.0.16/6", false},
		{"dual, but IPv6 mask too large.", "2001:db8::1/12,10.0.0.16/16", false},
		{"dual IPv4 IPv6", "10.0.0.16/12,2001:db8::/112", true},
		{"dual IPv6 IPv4", "2001:db8::/112,10.0.0.16/12", true},
	}
	for _, rt := range tests {

		actual := ValidateServiceSubnetSize(rt.subnet, nil)
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
		s        *kubeadmapi.APIEndpoint
		expected bool
	}{
		{
			name: "Valid IPv4 address / port",
			s: &kubeadmapi.APIEndpoint{
				AdvertiseAddress: "4.5.6.7",
				BindPort:         6443,
			},
			expected: true,
		},
		{
			name: "Valid IPv6 address / port",
			s: &kubeadmapi.APIEndpoint{
				AdvertiseAddress: "2001:db7::2",
				BindPort:         6443,
			},
			expected: true,
		},
		{
			name: "Invalid IPv4 address",
			s: &kubeadmapi.APIEndpoint{
				AdvertiseAddress: "1.2.34",
				BindPort:         6443,
			},
			expected: false,
		},
		{
			name: "Invalid IPv6 address",
			s: &kubeadmapi.APIEndpoint{
				AdvertiseAddress: "2001:db7:1",
				BindPort:         6443,
			},
			expected: false,
		},
		{
			name: "Invalid BindPort",
			s: &kubeadmapi.APIEndpoint{
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

func TestValidateCertificateKey(t *testing.T) {
	var tests = []struct {
		name           string
		certificateKey string
		expected       bool
	}{
		{
			name:           "Valid certificate key",
			certificateKey: "e6a2eb8581237ab72a4f494f30285ec12a9694d750b9785706a83bfcbbbd2204",
			expected:       true,
		},
		{
			name:           "Invalid hex encoded string",
			certificateKey: "z6a2eb8581237ab72a4f494f30285ec12a9694d750b9785706a83bfcbbbd2204",
			expected:       false,
		},
		{
			name:           "Invalid AES key size",
			certificateKey: "e6a2",
			expected:       false,
		},
	}
	for _, rt := range tests {
		actual := ValidateCertificateKey(rt.certificateKey, nil)
		t.Log(actual)
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

// TODO: Create a separated test for ValidateClusterConfiguration
func TestValidateInitConfiguration(t *testing.T) {
	nodename := "valid-nodename"
	criPath := fmt.Sprintf("%s:///some/path", kubeadmapiv1.DefaultContainerRuntimeURLScheme)
	var tests = []struct {
		name     string
		s        *kubeadmapi.InitConfiguration
		expected bool
	}{
		{"invalid missing InitConfiguration",
			&kubeadmapi.InitConfiguration{}, false},
		{"invalid missing token with IPv4 service subnet",
			&kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					Networking: kubeadmapi.Networking{
						ServiceSubnet: "10.96.0.1/12",
						DNSDomain:     "cluster.local",
					},
					CertificatesDir:     "/some/cert/dir",
					EncryptionAlgorithm: kubeadmapi.EncryptionAlgorithmRSA2048,
				},
				NodeRegistration: kubeadmapi.NodeRegistrationOptions{Name: nodename, CRISocket: criPath},
			}, false},
		{"invalid missing token with IPv6 service subnet",
			&kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					Networking: kubeadmapi.Networking{
						ServiceSubnet: "2001:db8::1/98",
						DNSDomain:     "cluster.local",
					},
					CertificatesDir:     "/some/cert/dir",
					EncryptionAlgorithm: kubeadmapi.EncryptionAlgorithmRSA2048,
				},
				NodeRegistration: kubeadmapi.NodeRegistrationOptions{Name: nodename, CRISocket: criPath},
			}, false},
		{"invalid missing node name",
			&kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					Networking: kubeadmapi.Networking{
						ServiceSubnet: "10.96.0.1/12",
						DNSDomain:     "cluster.local",
					},
					CertificatesDir:     "/some/other/cert/dir",
					EncryptionAlgorithm: kubeadmapi.EncryptionAlgorithmRSA2048,
				},
			}, false},
		{"valid InitConfiguration with incorrect IPv4 pod subnet",
			&kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					Networking: kubeadmapi.Networking{
						ServiceSubnet: "10.96.0.1/12",
						DNSDomain:     "cluster.local",
						PodSubnet:     "10.0.1.15",
					},
					CertificatesDir:     "/some/other/cert/dir",
					EncryptionAlgorithm: kubeadmapi.EncryptionAlgorithmRSA2048,
				},
				NodeRegistration: kubeadmapi.NodeRegistrationOptions{Name: nodename, CRISocket: criPath},
			}, false},
		{"valid InitConfiguration with IPv4 service subnet",
			&kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         6443,
				},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ImageRepository: "registry.k8s.io",
					Etcd: kubeadmapi.Etcd{
						Local: &kubeadmapi.LocalEtcd{
							DataDir: "/some/path",
						},
					},
					Networking: kubeadmapi.Networking{
						ServiceSubnet: "10.96.0.1/12",
						DNSDomain:     "cluster.local",
						PodSubnet:     "10.0.1.15/16",
					},
					CertificatesDir:     "/some/other/cert/dir",
					EncryptionAlgorithm: kubeadmapi.EncryptionAlgorithmRSA2048,
				},
				NodeRegistration: kubeadmapi.NodeRegistrationOptions{Name: nodename, CRISocket: criPath},
			}, true},
		{"valid InitConfiguration using IPv6 service subnet",
			&kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					AdvertiseAddress: "1:2:3::4",
					BindPort:         3446,
				},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ImageRepository: "registry.k8s.io",
					Etcd: kubeadmapi.Etcd{
						Local: &kubeadmapi.LocalEtcd{
							DataDir: "/some/path",
						},
					},
					Networking: kubeadmapi.Networking{
						ServiceSubnet: "2001:db8::1/112",
						DNSDomain:     "cluster.local",
					},
					CertificatesDir:     "/some/other/cert/dir",
					EncryptionAlgorithm: kubeadmapi.EncryptionAlgorithmECDSAP256,
				},
				NodeRegistration: kubeadmapi.NodeRegistrationOptions{Name: nodename, CRISocket: criPath},
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
	criPath := fmt.Sprintf("%s:///var/run/containerd/containerd.sock", kubeadmapiv1.DefaultContainerRuntimeURLScheme)
	var tests = []struct {
		s        *kubeadmapi.JoinConfiguration
		expected bool
	}{
		{&kubeadmapi.JoinConfiguration{}, false},
		{&kubeadmapi.JoinConfiguration{
			CACertPath: "/some/cert.crt",
			Discovery: kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token: "abcdef.1234567890123456@foobar",
				},
				File: &kubeadmapi.FileDiscovery{
					KubeConfigPath: "foo",
				},
			},
		}, false},
		{&kubeadmapi.JoinConfiguration{ // Pass without JoinControlPlane
			CACertPath: "/some/cert.crt",
			Discovery: kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token:             "abcdef.1234567890123456",
					APIServerEndpoint: "1.2.3.4:6443",
					CACertHashes:      []string{"aaaa"},
				},
				TLSBootstrapToken: "abcdef.1234567890123456",
			},
			NodeRegistration: kubeadmapi.NodeRegistrationOptions{
				Name:      "aaa",
				CRISocket: criPath,
			},
		}, true},
		{&kubeadmapi.JoinConfiguration{ // Pass with JoinControlPlane
			CACertPath: "/some/cert.crt",
			Discovery: kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token:             "abcdef.1234567890123456",
					APIServerEndpoint: "1.2.3.4:6443",
					CACertHashes:      []string{"aaaa"},
				},
				TLSBootstrapToken: "abcdef.1234567890123456",
			},
			NodeRegistration: kubeadmapi.NodeRegistrationOptions{
				Name:      "aaa",
				CRISocket: criPath,
			},
			ControlPlane: &kubeadmapi.JoinControlPlane{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					AdvertiseAddress: "1.2.3.4",
					BindPort:         1234,
				},
			},
		}, true},
		{&kubeadmapi.JoinConfiguration{ // Fail JoinControlPlane.AdvertiseAddress validation
			CACertPath: "/some/cert.crt",
			Discovery: kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token:             "abcdef.1234567890123456",
					APIServerEndpoint: "1.2.3.4:6443",
					CACertHashes:      []string{"aaaa"},
				},
				TLSBootstrapToken: "abcdef.1234567890123456",
			},
			NodeRegistration: kubeadmapi.NodeRegistrationOptions{
				Name:      "aaa",
				CRISocket: criPath,
			},
			ControlPlane: &kubeadmapi.JoinControlPlane{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
					AdvertiseAddress: "aaa",
					BindPort:         1234,
				},
			},
		}, false},
		{&kubeadmapi.JoinConfiguration{ // Fail JoinControlPlane.BindPort validation
			CACertPath: "/some/cert.crt",
			Discovery: kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token:             "abcdef.1234567890123456",
					APIServerEndpoint: "1.2.3.4:6443",
					CACertHashes:      []string{"aaaa"},
				},
				TLSBootstrapToken: "abcdef.1234567890123456",
			},
			NodeRegistration: kubeadmapi.NodeRegistrationOptions{
				Name:      "aaa",
				CRISocket: criPath,
			},
			ControlPlane: &kubeadmapi.JoinControlPlane{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{
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
		// Expected to succeed, --config is mixed with skip-* flags only or no other flags
		{[]string{"--config=hello", "--skip-token-print=true"}, true},
		{[]string{"--config=hello", "--ignore-preflight-errors=baz", "--skip-token-print"}, true},
		{[]string{"--config=hello", "--yes=true"}, true},
		// Expected to fail, --config is mixed with the --foo flag
		{[]string{"--config=hello", "--ignore-preflight-errors=baz", "--foo=bar"}, false},
		{[]string{"--config=hello", "--foo=bar"}, false},
		{[]string{"--config=hello", "--yes=true", "--foo=bar"}, false},
		// Expected to fail, --config is mixed with the upgrade related flag
		{[]string{"--config=hello", "--allow-experimental-upgrades"}, false},
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
		f.Bool("allow-experimental-upgrades", true, "upgrade flags for plan and apply command")
		f.Bool("skip-token-print", false, "flag not bound to config object")
		f.Bool("yes", false, "flag not bound to config object")
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
		expectedSet                         sets.Set[string]
		expectedError                       bool
	}{
		{ // empty lists in CLI and config file
			[]string{},
			[]string{},
			sets.New[string](),
			false,
		},
		{ // empty list in CLI only
			[]string{},
			[]string{"a"},
			sets.New("a"),
			false,
		},
		{ // empty list in config file only
			[]string{"a"},
			[]string{},
			sets.New("a"),
			false,
		},
		{ // no duplicates, no overlap
			[]string{"a", "b"},
			[]string{"c", "d"},
			sets.New("a", "b", "c", "d"),
			false,
		},
		{ // some duplicates, with some overlapping duplicates
			[]string{"a", "b", "a"},
			[]string{"c", "b"},
			sets.New("a", "b", "c"),
			false,
		},
		{ // empty list in CLI, but 'all' present in config file
			[]string{},
			[]string{"all"},
			sets.New("all"),
			false,
		},
		{ // empty list in config file, but 'all' present in CLI
			[]string{"all"},
			[]string{},
			sets.New("all"),
			false,
		},
		{ // some duplicates, only 'all' present in CLI and config file
			[]string{"all"},
			[]string{"all"},
			sets.New("all"),
			false,
		},
		{ // non-duplicate, but 'all' present together with individual checks in CLI
			[]string{"a", "b", "all"},
			[]string{},
			sets.New[string](),
			true,
		},
		{ // non-duplicate, but 'all' present together with individual checks in config file
			[]string{},
			[]string{"a", "b", "all"},
			sets.New[string](),
			true,
		},
		{ // non-duplicate, but 'all' present in config file, while values are in CLI, which is forbidden
			[]string{"a", "b"},
			[]string{"all"},
			sets.New[string](),
			true,
		},
		{ // non-duplicate, but 'all' present in CLI, while values are in config file, which is forbidden
			[]string{"all"},
			[]string{"a", "b"},
			sets.New[string](),
			true,
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
		d        *kubeadmapi.Discovery
		expected bool
	}{
		{
			"invalid: .BootstrapToken and .File cannot both be set",
			&kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token: "abcdef.1234567890123456",
				},
				File: &kubeadmapi.FileDiscovery{
					KubeConfigPath: "https://url/file.conf",
				},
			},
			false,
		},
		{
			"invalid: .BootstrapToken or .File must be set",
			&kubeadmapi.Discovery{},
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
		btd      *kubeadmapi.BootstrapTokenDiscovery
		expected bool
	}{
		{
			"invalid: .APIServerEndpoint not set",
			&kubeadmapi.BootstrapTokenDiscovery{
				Token: "abcdef.1234567890123456",
			},
			false,
		},
		{
			"invalid: using token-based discovery without .BootstrapToken.CACertHashes and .BootstrapToken.UnsafeSkipCAVerification",
			&kubeadmapi.BootstrapTokenDiscovery{
				Token:                    "abcdef.1234567890123456",
				APIServerEndpoint:        "192.168.122.100:6443",
				UnsafeSkipCAVerification: false,
			},
			false,
		},
		{
			"valid: using token-based discovery with .BootstrapToken.CACertHashes",
			&kubeadmapi.BootstrapTokenDiscovery{
				Token:                    "abcdef.1234567890123456",
				APIServerEndpoint:        "192.168.122.100:6443",
				CACertHashes:             []string{"sha256:7173b809ca12ec5dee4506cd86be934c4596dd234ee82c0662eac04a8c2c71dc"},
				UnsafeSkipCAVerification: false,
			},
			true,
		},
		{
			"valid: using token-based discovery with .BootstrapToken.CACertHashe but skip ca verification",
			&kubeadmapi.BootstrapTokenDiscovery{
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
	tmpfile, err := os.CreateTemp("/tmp", "test_discovery_file")
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

func TestValidateSocketPath(t *testing.T) {
	var tests = []struct {
		name           string
		criSocket      string
		expectedErrors bool
	}{
		{name: "valid socket URL", criSocket: kubeadmapiv1.DefaultContainerRuntimeURLScheme + "://" + "/some/path", expectedErrors: false},
		{name: "unsupported URL scheme", criSocket: "bla:///some/path", expectedErrors: true},
		{name: "missing URL scheme", criSocket: "/some/path", expectedErrors: true},
		{name: "unparseable URL", criSocket: ":::", expectedErrors: true},
		{name: "empty CRISocket", criSocket: "", expectedErrors: true},
	}
	for _, tc := range tests {
		actual := ValidateSocketPath(tc.criSocket, field.NewPath("criSocket"))
		actualErrors := len(actual) > 0
		if actualErrors != tc.expectedErrors {
			t.Errorf("error: socket path: %q\n\texpected: %t\n\t  actual: %t", tc.criSocket, tc.expectedErrors, actualErrors)
		}
	}
}

func TestValidateURLs(t *testing.T) {
	var tests = []struct {
		name           string
		urls           []string
		requireHTTPS   bool
		expectedErrors bool
	}{
		{
			name:           "valid urls (https not required)",
			urls:           []string{"http://example.com", "https://example.org"},
			requireHTTPS:   false,
			expectedErrors: false,
		},
		{
			name:           "valid urls (https required)",
			urls:           []string{"https://example.com", "https://example.org"},
			requireHTTPS:   true,
			expectedErrors: false,
		},
		{
			name:           "invalid url (https required)",
			urls:           []string{"http://example.com", "https://example.org"},
			requireHTTPS:   true,
			expectedErrors: true,
		},
		{
			name:           "URL parse error",
			urls:           []string{"::://example.com"},
			requireHTTPS:   false,
			expectedErrors: true,
		},
		{
			name:           "URL without scheme",
			urls:           []string{"example.com"},
			requireHTTPS:   false,
			expectedErrors: true,
		},
	}
	for _, tc := range tests {
		actual := ValidateURLs(tc.urls, tc.requireHTTPS, nil)
		actualErrors := len(actual) > 0
		if actualErrors != tc.expectedErrors {
			t.Errorf("error:\n\texpected: %t\n\t  actual: %t", tc.expectedErrors, actualErrors)
		}
	}
}

func TestValidateEtcd(t *testing.T) {
	var tests = []struct {
		name           string
		etcd           *kubeadmapi.Etcd
		expectedErrors bool
	}{
		{
			name:           "either .Etcd.Local or .Etcd.External is required",
			etcd:           &kubeadmapi.Etcd{},
			expectedErrors: true,
		},
		{
			name: ".Etcd.Local and .Etcd.External are mutually exclusive",
			etcd: &kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					DataDir: "/some/path",
				},
				External: &kubeadmapi.ExternalEtcd{
					Endpoints: []string{"10.100.0.1:2379", "10.100.0.2:2379"},
				},
			},
			expectedErrors: true,
		},
		{
			name: "either both or none of .Etcd.External.CertFile and .Etcd.External.KeyFile must be set",
			etcd: &kubeadmapi.Etcd{
				External: &kubeadmapi.ExternalEtcd{
					Endpoints: []string{"https://external.etcd1:2379", "https://external.etcd2:2379"},
					CertFile:  "/some/file.crt",
				},
			},
			expectedErrors: true,
		},
		{
			name: "setting .Etcd.External.CertFile and .Etcd.External.KeyFile requires .Etcd.External.CAFile",
			etcd: &kubeadmapi.Etcd{
				External: &kubeadmapi.ExternalEtcd{
					Endpoints: []string{"https://external.etcd1:2379", "https://external.etcd2:2379"},
					CertFile:  "/some/file.crt",
					KeyFile:   "/some/file.key",
				},
			},
			expectedErrors: true,
		},
		{
			name: "valid external etcd",
			etcd: &kubeadmapi.Etcd{
				External: &kubeadmapi.ExternalEtcd{
					Endpoints: []string{"https://external.etcd1:2379", "https://external.etcd2:2379"},
					CertFile:  "/etcd.crt",
					KeyFile:   "/etcd.key",
					CAFile:    "/etcd-ca.crt",
				},
			},
			expectedErrors: false,
		},
		{
			name: "valid external etcd (no TLS)",
			etcd: &kubeadmapi.Etcd{
				External: &kubeadmapi.ExternalEtcd{
					Endpoints: []string{"http://10.100.0.1:2379", "http://10.100.0.2:2379"},
				},
			},
			expectedErrors: false,
		},
	}

	for _, tc := range tests {
		actual := ValidateEtcd(tc.etcd, field.NewPath("etcd"))
		actualErrors := len(actual) > 0
		if actualErrors != tc.expectedErrors {
			t.Errorf("Error: \n\texpected: %t\n\t  actual: %t\n\t  encountered errors: %v",
				tc.expectedErrors,
				actualErrors,
				actual,
			)
		}
	}
}

func TestValidateEncryptionAlgorithm(t *testing.T) {
	var tests = []struct {
		name           string
		algo           kubeadmapi.EncryptionAlgorithmType
		expectedErrors bool
	}{
		{name: "valid RSA-2048", algo: kubeadmapi.EncryptionAlgorithmRSA2048, expectedErrors: false},
		{name: "valid RSA-3072", algo: kubeadmapi.EncryptionAlgorithmRSA3072, expectedErrors: false},
		{name: "valid RSA-4096", algo: kubeadmapi.EncryptionAlgorithmRSA4096, expectedErrors: false},
		{name: "valid ECDSA-P256", algo: kubeadmapi.EncryptionAlgorithmECDSAP256, expectedErrors: false},
		{name: "invalid algorithm", algo: "foo", expectedErrors: true},
		{name: "empty algorithm returns an error", algo: "", expectedErrors: true},
	}
	for _, tc := range tests {
		actual := ValidateEncryptionAlgorithm(tc.algo, field.NewPath("encryptionAlgorithm"))
		actualErrors := len(actual) > 0
		if actualErrors != tc.expectedErrors {
			t.Errorf("error: validate public key algorithm: %q\n\texpected: %t\n\t  actual: %t", tc.algo, tc.expectedErrors, actualErrors)
		}
	}
}

func TestGetClusterNodeMask(t *testing.T) {
	tests := []struct {
		name          string
		cfg           *kubeadmapi.ClusterConfiguration
		isIPv6        bool
		expectedMask  int
		expectedError bool
	}{
		{
			name:         "dual ipv4 default mask",
			cfg:          &kubeadmapi.ClusterConfiguration{},
			isIPv6:       false,
			expectedMask: 24,
		},
		{
			name: "dual ipv4 custom mask",
			cfg: &kubeadmapi.ClusterConfiguration{
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "node-cidr-mask-size", Value: "21"},
						{Name: "node-cidr-mask-size-ipv4", Value: "23"},
					},
				},
			},
			isIPv6:       false,
			expectedMask: 23,
		},
		{
			name:         "dual ipv6 default mask",
			cfg:          &kubeadmapi.ClusterConfiguration{},
			isIPv6:       true,
			expectedMask: 64,
		},
		{
			name: "dual ipv6 custom mask",
			cfg: &kubeadmapi.ClusterConfiguration{
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "node-cidr-mask-size-ipv6", Value: "83"},
					},
				},
			},
			isIPv6:       true,
			expectedMask: 83,
		},
		{
			name: "dual ipv4 custom mask",
			cfg: &kubeadmapi.ClusterConfiguration{
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "node-cidr-mask-size-ipv4", Value: "23"},
					},
				},
			},
			isIPv6:       false,
			expectedMask: 23,
		},
		{
			name: "dual ipv4 wrong mask",
			cfg: &kubeadmapi.ClusterConfiguration{
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "node-cidr-mask-size-ipv4", Value: "aa"},
					},
				},
			},
			isIPv6:        false,
			expectedError: true,
		},
		{
			name: "dual ipv6 default mask and legacy flag",
			cfg: &kubeadmapi.ClusterConfiguration{
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "node-cidr-mask-size", Value: "23"},
					},
				},
			},
			isIPv6:       true,
			expectedMask: 64,
		},
		{
			name: "dual ipv6 custom mask and legacy flag",
			cfg: &kubeadmapi.ClusterConfiguration{
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "node-cidr-mask-size", Value: "23"},
						{Name: "node-cidr-mask-size-ipv6", Value: "83"},
					},
				},
			},
			isIPv6:       true,
			expectedMask: 83,
		},
		{
			name: "dual ipv6 custom mask and wrong flag",
			cfg: &kubeadmapi.ClusterConfiguration{
				ControllerManager: kubeadmapi.ControlPlaneComponent{
					ExtraArgs: []kubeadmapi.Arg{
						{Name: "node-cidr-mask-size", Value: "23"},
						{Name: "node-cidr-mask-size-ipv6", Value: "a83"},
					},
				},
			},
			isIPv6:        true,
			expectedError: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mask, err := getClusterNodeMask(test.cfg, test.isIPv6)
			if (err == nil) == test.expectedError {
				t.Errorf("expected error: %v, got %v", test.expectedError, err)
			}
			if mask != test.expectedMask {
				t.Errorf("expected mask: %d, got %d", test.expectedMask, mask)
			}
		})
	}
}

func TestValidateImageRepository(t *testing.T) {
	var tests = []struct {
		imageRepository string
		expectedErrors  bool
	}{
		{
			imageRepository: "a",
			expectedErrors:  false,
		},
		{
			imageRepository: "a.b.c",
			expectedErrors:  false,
		},
		{
			imageRepository: "a.b.c/repo",
			expectedErrors:  false,
		},
		{
			imageRepository: "a:5000",
			expectedErrors:  false,
		},
		{
			imageRepository: "a.b.c:5000",
			expectedErrors:  false,
		},
		{
			imageRepository: "a.b.c:5000/repo",
			expectedErrors:  false,
		},
		{
			imageRepository: "a/b/c",
			expectedErrors:  false,
		},
		{
			imageRepository: "127.0.0.1:5000/repo",
			expectedErrors:  false,
		},
		{
			imageRepository: "",
			expectedErrors:  true,
		},
		{
			imageRepository: `a.b/c
			s`,
			expectedErrors: true,
		},
		{
			imageRepository: " a.b.c",
			expectedErrors:  true,
		},
		{
			imageRepository: "a.b c",
			expectedErrors:  true,
		},
		{
			imageRepository: "a.b.c:5000/",
			expectedErrors:  true,
		},
		{
			imageRepository: "https://a.b.c:5000",
			expectedErrors:  true,
		},
		{
			imageRepository: "a//b/c",
			expectedErrors:  true,
		},
		{
			imageRepository: "a.b.c:5000/test:1.0",
			expectedErrors:  true,
		},
	}

	for _, tc := range tests {
		actual := ValidateImageRepository(tc.imageRepository, nil)
		actualErrors := len(actual) > 0
		if actualErrors != tc.expectedErrors {
			t.Errorf("case %q error:\n\t expected: %t\n\t actual: %t", tc.imageRepository, tc.expectedErrors, actualErrors)
		}
	}
}

func TestValidateAbsolutePath(t *testing.T) {
	var tests = []struct {
		name           string
		path           string
		expectedErrors bool
	}{
		{name: "valid absolute path", path: "/etc/cert/dir", expectedErrors: false},
		{name: "relative path", path: "./tmp", expectedErrors: true},
		{name: "invalid path", path: "foo..", expectedErrors: true},
	}
	for _, tc := range tests {
		actual := ValidateAbsolutePath(tc.path, field.NewPath("certificatesDir"))
		actualErrors := len(actual) > 0
		if actualErrors != tc.expectedErrors {
			t.Errorf("error: validate absolute path: %q\n\texpected: %t\n\t  actual: %t", tc.path, tc.expectedErrors, actualErrors)
		}
	}
}

func TestValidateExtraArgs(t *testing.T) {
	var tests = []struct {
		name           string
		args           []kubeadmapi.Arg
		expectedErrors int
	}{
		{
			name:           "valid argument",
			args:           []kubeadmapi.Arg{{Name: "foo", Value: "bar"}},
			expectedErrors: 0,
		},
		{
			name:           "invalid one argument",
			args:           []kubeadmapi.Arg{{Name: "", Value: "bar"}},
			expectedErrors: 1,
		},
		{
			name:           "invalid two arguments",
			args:           []kubeadmapi.Arg{{Name: "", Value: "foo"}, {Name: "", Value: "bar"}},
			expectedErrors: 2,
		},
	}

	for _, tc := range tests {
		actual := ValidateExtraArgs(tc.args, nil)
		if len(actual) != tc.expectedErrors {
			t.Errorf("case %q:\n\t expected errors: %v\n\t got: %v\n\t errors: %v", tc.name, tc.expectedErrors, len(actual), actual)
		}
	}
}

func TestValidateUnmountFlags(t *testing.T) {
	var tests = []struct {
		name           string
		flags          []string
		expectedErrors int
	}{
		{
			name:           "nil input",
			flags:          nil,
			expectedErrors: 0,
		},
		{
			name: "all valid flags",
			flags: []string{
				kubeadmapi.UnmountFlagMNTForce,
				kubeadmapi.UnmountFlagMNTDetach,
				kubeadmapi.UnmountFlagMNTExpire,
				kubeadmapi.UnmountFlagUmountNoFollow,
			},
			expectedErrors: 0,
		},
		{
			name: "invalid two flags",
			flags: []string{
				"foo",
				"bar",
			},
			expectedErrors: 2,
		},
	}

	for _, tc := range tests {
		actual := ValidateUnmountFlags(tc.flags, nil)
		if len(actual) != tc.expectedErrors {
			t.Errorf("case %q:\n\t expected errors: %v\n\t got: %v\n\t errors: %v", tc.name, tc.expectedErrors, len(actual), actual)
		}
	}
}

func TestPullPolicy(t *testing.T) {
	var tests = []struct {
		name           string
		policy         string
		expectedErrors int
	}{
		{
			name:           "empty policy causes no errors", // gets defaulted
			policy:         "",
			expectedErrors: 0,
		},
		{
			name:           "invalid policy",
			policy:         "foo",
			expectedErrors: 1,
		},
		{
			name:           "valid policy",
			policy:         "IfNotPresent",
			expectedErrors: 0,
		},
	}

	for _, tc := range tests {
		actual := ValidateImagePullPolicy(corev1.PullPolicy(tc.policy), nil)
		if len(actual) != tc.expectedErrors {
			t.Errorf("case %q:\n\t expected errors: %v\n\t got: %v\n\t errors: %v", tc.name, tc.expectedErrors, len(actual), actual)
		}
	}
}
