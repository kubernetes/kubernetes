/*
Copyright 2016 The Kubernetes Authors.

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

package certs

import (
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"testing"

	certutil "k8s.io/client-go/util/cert"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestCreatePKIAssets(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	var tests = []struct {
		cfg      *kubeadmapi.MasterConfiguration
		expected bool
	}{
		{
			cfg:      &kubeadmapi.MasterConfiguration{},
			expected: false,
		},
		{
			// CIDR too small
			cfg: &kubeadmapi.MasterConfiguration{
				API:             kubeadmapi.API{AdvertiseAddress: "1.2.3.4"},
				Networking:      kubeadmapi.Networking{ServiceSubnet: "10.0.0.1/1"},
				CertificatesDir: fmt.Sprintf("%s/etc/kubernetes/pki", tmpdir),
			},
			expected: false,
		},
		{
			// CIDR invalid
			cfg: &kubeadmapi.MasterConfiguration{
				API:             kubeadmapi.API{AdvertiseAddress: "1.2.3.4"},
				Networking:      kubeadmapi.Networking{ServiceSubnet: "invalid"},
				CertificatesDir: fmt.Sprintf("%s/etc/kubernetes/pki", tmpdir),
			},
			expected: false,
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:             kubeadmapi.API{AdvertiseAddress: "1.2.3.4"},
				Networking:      kubeadmapi.Networking{ServiceSubnet: "10.0.0.1/24"},
				CertificatesDir: fmt.Sprintf("%s/etc/kubernetes/pki", tmpdir),
			},
			expected: true,
		},
	}
	for _, rt := range tests {
		actual := CreatePKIAssets(rt.cfg)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed CreatePKIAssets with an error:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
		}
	}
}

func TestCheckAltNamesExist(t *testing.T) {
	var tests = []struct {
		IPs              []net.IP
		DNSNames         []string
		requiredAltNames certutil.AltNames
		succeed          bool
	}{
		{
			// equal
			requiredAltNames: certutil.AltNames{IPs: []net.IP{net.ParseIP("1.1.1.1"), net.ParseIP("192.168.1.2")}, DNSNames: []string{"foo", "bar", "baz"}},
			IPs:              []net.IP{net.ParseIP("1.1.1.1"), net.ParseIP("192.168.1.2")},
			DNSNames:         []string{"foo", "bar", "baz"},
			succeed:          true,
		},
		{
			// the loaded cert has more ips than required, ok
			requiredAltNames: certutil.AltNames{IPs: []net.IP{net.ParseIP("1.1.1.1"), net.ParseIP("192.168.1.2")}, DNSNames: []string{"foo", "bar", "baz"}},
			IPs:              []net.IP{net.ParseIP("192.168.2.5"), net.ParseIP("1.1.1.1"), net.ParseIP("192.168.1.2")},
			DNSNames:         []string{"a", "foo", "b", "bar", "baz"},
			succeed:          true,
		},
		{
			// the loaded cert doesn't have all ips
			requiredAltNames: certutil.AltNames{IPs: []net.IP{net.ParseIP("1.1.1.1"), net.ParseIP("192.168.2.5"), net.ParseIP("192.168.1.2")}, DNSNames: []string{"foo", "bar", "baz"}},
			IPs:              []net.IP{net.ParseIP("1.1.1.1"), net.ParseIP("192.168.1.2")},
			DNSNames:         []string{"foo", "bar", "baz"},
			succeed:          false,
		},
		{
			// the loaded cert doesn't have all ips
			requiredAltNames: certutil.AltNames{IPs: []net.IP{net.ParseIP("1.1.1.1"), net.ParseIP("192.168.1.2")}, DNSNames: []string{"foo", "bar", "b", "baz"}},
			IPs:              []net.IP{net.ParseIP("1.1.1.1"), net.ParseIP("192.168.1.2")},
			DNSNames:         []string{"foo", "bar", "baz"},
			succeed:          false,
		},
	}

	for _, rt := range tests {
		succeeded := checkAltNamesExist(rt.IPs, rt.DNSNames, rt.requiredAltNames)
		if succeeded != rt.succeed {
			t.Errorf(
				"failed checkAltNamesExist:\n\texpected: %t\n\t  actual: %t",
				rt.succeed,
				succeeded,
			)
		}
	}
}

func TestGetAltNames(t *testing.T) {
	var tests = []struct {
		cfgaltnames      []string
		hostname         string
		dnsdomain        string
		servicecidr      string
		expectedIPs      []string
		expectedDNSNames []string
	}{
		{
			cfgaltnames:      []string{"foo", "192.168.200.1", "bar.baz"},
			hostname:         "my-node",
			dnsdomain:        "cluster.external",
			servicecidr:      "10.96.0.1/12",
			expectedIPs:      []string{"192.168.200.1", "10.96.0.1"},
			expectedDNSNames: []string{"my-node", "kubernetes", "kubernetes.default", "kubernetes.default.svc", "kubernetes.default.svc.cluster.external", "foo", "bar.baz"},
		},
	}

	for _, rt := range tests {
		_, svcSubnet, _ := net.ParseCIDR(rt.servicecidr)
		actual := getAltNames(rt.cfgaltnames, rt.hostname, rt.dnsdomain, svcSubnet)
		for i := range actual.IPs {
			if rt.expectedIPs[i] != actual.IPs[i].String() {
				t.Errorf(
					"failed getAltNames:\n\texpected: %s\n\t  actual: %s",
					rt.expectedIPs[i],
					actual.IPs[i].String(),
				)
			}
		}
		for i := range actual.DNSNames {
			if rt.expectedDNSNames[i] != actual.DNSNames[i] {
				t.Errorf(
					"failed getAltNames:\n\texpected: %s\n\t  actual: %s",
					rt.expectedDNSNames[i],
					actual.DNSNames[i],
				)
			}
		}
	}
}
