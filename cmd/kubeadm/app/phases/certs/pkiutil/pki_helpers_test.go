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

package pkiutil

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"io/ioutil"
	"os"
	"testing"

	certutil "k8s.io/client-go/util/cert"
)

func TestNewCertificateAuthority(t *testing.T) {
	cert, key, err := NewCertificateAuthority()

	if cert == nil {
		t.Errorf(
			"failed NewCertificateAuthority, cert == nil",
		)
	}
	if key == nil {
		t.Errorf(
			"failed NewCertificateAuthority, key == nil",
		)
	}
	if err != nil {
		t.Errorf(
			"failed NewCertificateAuthority with an error: %v",
			err,
		)
	}
}

func TestNewCertAndKey(t *testing.T) {
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
			// Should succeed
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
		config := certutil.Config{
			CommonName:   "test",
			Organization: []string{"test"},
			Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		}
		_, _, actual := NewCertAndKey(caCert, caKey, config)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed NewCertAndKey:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
		}
	}
}

func TestWriteCertAndKey(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	caKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("Couldn't create rsa Private Key")
	}
	caCert := &x509.Certificate{}
	actual := WriteCertAndKey(tmpdir, "foo", caCert, caKey)
	if actual != nil {
		t.Errorf(
			"failed WriteCertAndKey with an error: %v",
			actual,
		)
	}
}

func TestCertOrKeyExist(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	caKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("Couldn't create rsa Private Key")
	}
	caCert := &x509.Certificate{}
	actual := WriteCertAndKey(tmpdir, "foo", caCert, caKey)
	if actual != nil {
		t.Errorf(
			"failed WriteCertAndKey with an error: %v",
			actual,
		)
	}

	var tests = []struct {
		path     string
		name     string
		expected bool
	}{
		{
			path:     "",
			name:     "",
			expected: false,
		},
		{
			path:     tmpdir,
			name:     "foo",
			expected: true,
		},
	}
	for _, rt := range tests {
		actual := CertOrKeyExist(rt.path, rt.name)
		if actual != rt.expected {
			t.Errorf(
				"failed CertOrKeyExist:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				actual,
			)
		}
	}
}

func TestTryLoadCertAndKeyFromDisk(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	caCert, caKey, err := NewCertificateAuthority()
	if err != nil {
		t.Errorf(
			"failed to create cert and key with an error: %v",
			err,
		)
	}
	err = WriteCertAndKey(tmpdir, "foo", caCert, caKey)
	if err != nil {
		t.Errorf(
			"failed to write cert and key with an error: %v",
			err,
		)
	}

	var tests = []struct {
		path     string
		name     string
		expected bool
	}{
		{
			path:     "",
			name:     "",
			expected: false,
		},
		{
			path:     tmpdir,
			name:     "foo",
			expected: true,
		},
	}
	for _, rt := range tests {
		_, _, actual := TryLoadCertAndKeyFromDisk(rt.path, rt.name)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed TryLoadCertAndKeyFromDisk:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
		}
	}
}
