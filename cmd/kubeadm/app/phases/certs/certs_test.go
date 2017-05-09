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
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
)

// TODO: Integration test cases
// valid ca.{crt,key} exists =>  => do nothing
// invalid ca.{crt,key} exists => error
// only one of the .crt or .key file exists => error

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

func TestPKIAssetsAttributes(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir: %v", err)
	}
	defer os.RemoveAll(tmpdir)

	// Create all PKI assets
	defaultIP := "1.2.3.4"
	_, err = CreatePKIAssets(&kubeadmapi.MasterConfiguration{
		CertificatesDir: tmpdir,
		API:             kubeadmapi.API{AdvertiseAddress: defaultIP},
		Networking:      kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
	})
	if err != nil {
		t.Fatalf("Error creating CreatePKIAssets: %v", err)
	}

	// Assert CA.crt properties (isCA)
	CAcert, err := pkiutil.TryLoadCertFromDisk(tmpdir, constants.CACertAndKeyBaseName)
	if err != nil {
		t.Fatalf("Error loading CACert: %v", err)
	}
	if !CAcert.IsCA {
		t.Error("CACert is not a validaCA")
	}

	// Assert apiserver.crt properties (signed from CA, server authorithy for expected DNSNames/IPAddresses)
	APIserverCert, err := pkiutil.TryLoadCertFromDisk(tmpdir, constants.APIServerCertAndKeyBaseName)
	if err != nil {
		t.Fatalf("Error loading APIserverCert: %v", err)
	}
	if err := APIserverCert.CheckSignatureFrom(CAcert); err != nil {
		t.Error("APIserverCert is not signed by CA")
	}
	if len(APIserverCert.ExtKeyUsage) != 1 || APIserverCert.ExtKeyUsage[0] != x509.ExtKeyUsageServerAuth {
		t.Error("APIserverCert is not a server authority")
	}

	hostname, err := os.Hostname()
	if err != nil {
		t.Errorf("couldn't get the hostname: %v", err)
	}
	for i, name := range []string{hostname, "kubernetes", "kubernetes.default", "kubernetes.default.svc", "kubernetes.default.svc.cluster.local"} {
		if APIserverCert.DNSNames[i] != name {
			t.Errorf("APIserverCert.DNSNames[%d] is %s instead of %s", i, APIserverCert.DNSNames[i], name)
		}
	}
	for i, ip := range []string{"10.96.0.1", defaultIP} {
		if APIserverCert.IPAddresses[i].String() != ip {
			t.Errorf("APIserverCert.IPAddresses[%d] is %s instead of %s", i, APIserverCert.IPAddresses[i], ip)
		}
	}

	// Assert apiserver-kubelet-client.crt properties (signed from CA, client authorithy for expected organization)
	APIServerKubeletClientCert, err := pkiutil.TryLoadCertFromDisk(tmpdir, constants.APIServerKubeletClientCertAndKeyBaseName)
	if err != nil {
		t.Fatalf("Error loading APIServerKubeletClientCert: %v", err)
	}
	if err := APIServerKubeletClientCert.CheckSignatureFrom(CAcert); err != nil {
		t.Error("APIServerKubeletClientCert is not signed by CA")
	}
	if len(APIServerKubeletClientCert.ExtKeyUsage) != 1 || APIServerKubeletClientCert.ExtKeyUsage[0] != x509.ExtKeyUsageClientAuth {
		t.Error("APIServerKubeletClientCert is not a client authority")
	}
	if len(APIServerKubeletClientCert.Subject.Organization) != 1 || APIServerKubeletClientCert.Subject.Organization[0] != constants.MastersGroup {
		t.Errorf("APIServerKubeletClientCert is not part of %s organization", constants.MastersGroup)
	}

	//TODO: SA

	// Assert front-proxy-ca.crt properties (isCA)
	FrontProxyCA, err := pkiutil.TryLoadCertFromDisk(tmpdir, constants.FrontProxyCACertAndKeyBaseName)
	if err != nil {
		t.Fatalf("Error loading FrontProxyCA: %v", err)
	}
	if !FrontProxyCA.IsCA {
		t.Error("FrontProxyCA is not a valid CA")
	}

	// Assert front-proxy-client.crt properties (signed from CA, client authorithy)
	FrontProxyClientCert, err := pkiutil.TryLoadCertFromDisk(tmpdir, constants.FrontProxyClientCertAndKeyBaseName)
	if err != nil {
		t.Fatalf("Error loading FrontProxyClientCert: %v", err)
	}
	if err := FrontProxyClientCert.CheckSignatureFrom(FrontProxyCA); err != nil {
		t.Error("FrontProxyClientCert is not signed by FrontProxyCA")
	}
	if len(FrontProxyClientCert.ExtKeyUsage) != 1 || FrontProxyClientCert.ExtKeyUsage[0] != x509.ExtKeyUsageClientAuth {
		t.Error("FrontProxyClientCert is not a client authority")
	}
}

func TestCreateOrUseExisting(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir: %v", err)
	}
	defer os.RemoveAll(tmpdir)

	var tests = []CreateCertFunc{
		CreateCACertAndKey,
		CreateAPIServerCertAndKey,
		CreateAPIServerKubeletClientCertAndKey,
		CreateServiceAccountKeyAndPublicKey,
		CreateFrontProxyCACertAndKey,
		CreateFrontProxyClientCertAndKey,
	}

	var cfg = &kubeadmapi.MasterConfiguration{
		API:             kubeadmapi.API{AdvertiseAddress: "1.2.3.4"},
		Networking:      kubeadmapi.Networking{ServiceSubnet: "10.0.0.1/24"},
		CertificatesDir: fmt.Sprintf("%s/etc/kubernetes/pki", tmpdir),
	}

	for i, createCertFunc := range tests {
		// first run > create
		r, err := createCertFunc(cfg)
		if err != nil {
			t.Errorf("failed createCertFunc[%d] with an error: %v", i, err)
		}
		if r.UsedExistingCert != false {
			t.Errorf("createCertFunc[%d] returned UsedExistingCert=true, expected false", i)
		}
		// second run > use existing
		r, err = createCertFunc(cfg)
		if err != nil {
			t.Errorf("failed createCertFunc[%d] with an error: %v", i, err)
		}
		if r.UsedExistingCert != true {
			t.Errorf("createCertFunc[%d] returned UsedExistingCert=false, expected true", i)
		}
	}
}

func TestCreatePKIAssets(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir: %v", err)
	}
	defer os.RemoveAll(tmpdir)

	var tests = []struct {
		cfg      *kubeadmapi.MasterConfiguration
		expected bool
	}{
		{
			// Invalid AdvertiseAddress & ServiceSubnet
			cfg: &kubeadmapi.MasterConfiguration{
				CertificatesDir: fmt.Sprintf("%s/etc/kubernetes/pki", tmpdir),
			},
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
		_, actual := CreatePKIAssets(rt.cfg)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed CreatePKIAssets with an error:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
		}
	}
}
