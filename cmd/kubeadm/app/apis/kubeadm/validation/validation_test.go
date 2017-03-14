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

func TestValidateAuthorizationMode(t *testing.T) {
	var tests = []struct {
		s        string
		f        *field.Path
		expected bool
	}{
		{"", nil, false},
		{"rBAC", nil, false},      // not supported
		{"not valid", nil, false}, // not supported
		{"RBAC", nil, true},       // supported
		{"Webhook", nil, true},    // supported
	}
	for _, rt := range tests {
		actual := ValidateAuthorizationMode(rt.s, rt.f)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"failed ValidateAuthorizationMode:\n\texpected: %t\n\t  actual: %t",
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
		{[]string{}, true},                                                  // ok if not provided
		{[]string{"1,2,,3"}, false},                                         // not a DNS label or IP
		{[]string{"my-hostname", "???&?.garbage"}, false},                   // not valid
		{[]string{"my-hostname", "my.subdomain", "1.2.3.4"}, true},          // supported
		{[]string{"my-hostname2", "my.other.subdomain", "10.0.0.10"}, true}, // supported
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
		ip       string
		expected bool
	}{
		{"", false},           // not valid
		{"1234", false},       // not valid
		{"1.2", false},        // not valid
		{"1.2.3.4/16", false}, // not valid
		{"1.2.3.4", true},     // valid
		{"16.0.1.1", true},    // valid
	}
	for _, rt := range tests {
		actual := ValidateIPFromString(rt.ip, nil)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"failed ValidateIPFromString:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}

func TestValidateIPNetFromString(t *testing.T) {
	var tests = []struct {
		subnet   string
		minaddrs int64
		expected bool
	}{
		{"", 0, false},              // not valid
		{"1234", 0, false},          // not valid
		{"abc", 0, false},           // not valid
		{"1.2.3.4", 0, false},       // ip not valid
		{"10.0.0.16/29", 10, false}, // valid, but too small. At least 10 addrs needed
		{"10.0.0.16/12", 10, true},  // valid
	}
	for _, rt := range tests {
		actual := ValidateIPNetFromString(rt.subnet, rt.minaddrs, nil)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"failed ValidateIPNetFromString:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(len(actual) == 0),
			)
		}
	}
}

func TestValidateMasterConfiguration(t *testing.T) {
	var tests = []struct {
		s        *kubeadm.MasterConfiguration
		expected bool
	}{
		{&kubeadm.MasterConfiguration{}, false},
		{&kubeadm.MasterConfiguration{
			AuthorizationMode: "RBAC",
			Networking: kubeadm.Networking{
				ServiceSubnet: "10.96.0.1/12",
				DNSDomain:     "cluster.local",
			},
			CertificatesDir: "/some/cert/dir",
		}, false},
		{&kubeadm.MasterConfiguration{
			AuthorizationMode: "RBAC",
			Networking: kubeadm.Networking{
				ServiceSubnet: "10.96.0.1/12",
				DNSDomain:     "cluster.local",
			},
			CertificatesDir: "/some/other/cert/dir",
			Token:           "abcdef.0123456789abcdef",
		}, true},
	}
	for _, rt := range tests {
		actual := ValidateMasterConfiguration(rt.s)
		if (len(actual) == 0) != rt.expected {
			t.Errorf(
				"failed ValidateMasterConfiguration:\n\texpected: %t\n\t  actual: %t",
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
