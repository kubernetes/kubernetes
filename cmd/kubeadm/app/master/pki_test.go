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

package master

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"os"
	"testing"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	certutil "k8s.io/kubernetes/pkg/util/cert"
)

func TestNewCertificateAuthority(t *testing.T) {
	r, x, err := newCertificateAuthority()

	if r == nil {
		t.Errorf(
			"failed newCertificateAuthority, rsa key == nil",
		)
	}
	if x == nil {
		t.Errorf(
			"failed newCertificateAuthority, x509 cert == nil",
		)
	}
	if err != nil {
		t.Errorf(
			"failed newCertificateAuthority with an error: %v",
			err,
		)
	}
}

func TestNewServerKeyAndCert(t *testing.T) {
	var tests = []struct {
		cfg       *kubeadmapi.MasterConfiguration
		caKeySize int
		expected  bool
	}{
		{
			// given CIDR too small
			cfg: &kubeadmapi.MasterConfiguration{
				Networking: kubeadm.Networking{ServiceSubnet: "10.0.0.1/1"},
			},
			caKeySize: 2048,
			expected:  false,
		},
		{
			// bad CIDR
			cfg: &kubeadmapi.MasterConfiguration{
				Networking: kubeadm.Networking{ServiceSubnet: "foo"},
			},
			caKeySize: 2048,
			expected:  false,
		},
		{
			// RSA key too small
			cfg: &kubeadmapi.MasterConfiguration{
				Networking: kubeadm.Networking{ServiceSubnet: "10.0.0.1/24"},
			},
			caKeySize: 128,
			expected:  false,
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				Networking: kubeadm.Networking{ServiceSubnet: "10.0.0.1/24"},
			},
			caKeySize: 2048,
			expected:  true,
		},
	}

	for _, rt := range tests {
		caKey, err := rsa.GenerateKey(rand.Reader, rt.caKeySize)
		if err != nil {
			t.Fatalf("Couldn't create rsa Private Key")
		}
		caCert := &x509.Certificate{}
		altNames := certutil.AltNames{}
		_, _, actual := newServerKeyAndCert(rt.cfg, caCert, caKey, altNames)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed newServerKeyAndCert:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
		}
	}
}

func TestNewClientKeyAndCert(t *testing.T) {
	var tests = []struct {
		caKeySize int
		expected  bool
	}{
		{
			// RSA key too small
			caKeySize: 128,
			expected:  false,
		},
		{
			caKeySize: 2048,
			expected:  true,
		},
	}

	for _, rt := range tests {
		caKey, err := rsa.GenerateKey(rand.Reader, rt.caKeySize)
		if err != nil {
			t.Fatalf("Couldn't create rsa Private Key")
		}
		caCert := &x509.Certificate{}
		_, _, actual := newClientKeyAndCert(caCert, caKey)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed newClientKeyAndCert:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
		}
	}
}

func TestWriteKeysAndCert(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.Remove(tmpdir)

	caKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("Couldn't create rsa Private Key")
	}
	caCert := &x509.Certificate{}
	actual := writeKeysAndCert(tmpdir, "foo", caKey, caCert)
	if actual != nil {
		t.Errorf(
			"failed writeKeysAndCert with an error: %v",
			actual,
		)
	}
}

func TestPathsKeysCerts(t *testing.T) {
	var tests = []struct {
		pkiPath  string
		name     string
		expected []string
	}{
		{
			pkiPath:  "foo",
			name:     "bar",
			expected: []string{"foo/bar-pub.pem", "foo/bar-key.pem", "foo/bar.pem"},
		},
		{
			pkiPath:  "bar",
			name:     "foo",
			expected: []string{"bar/foo-pub.pem", "bar/foo-key.pem", "bar/foo.pem"},
		},
	}

	for _, rt := range tests {
		a, b, c := pathsKeysCerts(rt.pkiPath, rt.name)
		all := []string{a, b, c}
		for i := range all {
			if all[i] != rt.expected[i] {
				t.Errorf(
					"failed pathsKeysCerts:\n\texpected: %s\n\t  actual: %s",
					rt.expected[i],
					all[i],
				)
			}
		}
	}
}

func TestNewServiceAccountKey(t *testing.T) {
	r, err := newServiceAccountKey()
	if r == nil {
		t.Errorf(
			"failed newServiceAccountKey, rsa key == nil",
		)
	}
	if err != nil {
		t.Errorf(
			"failed newServiceAccountKey with an error: %v",
			err,
		)
	}
}

func TestCreatePKIAssets(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.Remove(tmpdir)

	// set up tmp GlobalEnvParams values for testing
	oldEnv := kubeadmapi.GlobalEnvParams
	kubeadmapi.GlobalEnvParams.HostPKIPath = fmt.Sprintf("%s/etc/kubernetes/pki", tmpdir)
	defer func() { kubeadmapi.GlobalEnvParams = oldEnv }()

	var tests = []struct {
		cfg      *kubeadmapi.MasterConfiguration
		expected bool
	}{
		{
			cfg:      &kubeadmapi.MasterConfiguration{},
			expected: false,
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:        kubeadm.API{AdvertiseAddresses: []string{"10.0.0.1"}},
				Networking: kubeadm.Networking{ServiceSubnet: "10.0.0.1/1"},
			},
			expected: false,
		},
		{
			cfg: &kubeadmapi.MasterConfiguration{
				API:        kubeadm.API{AdvertiseAddresses: []string{"10.0.0.1"}},
				Networking: kubeadm.Networking{ServiceSubnet: "10.0.0.1/24"},
			},
			expected: true,
		},
	}
	for _, rt := range tests {
		_, _, actual := CreatePKIAssets(rt.cfg)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed CreatePKIAssets with an error:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
		}
	}
}
