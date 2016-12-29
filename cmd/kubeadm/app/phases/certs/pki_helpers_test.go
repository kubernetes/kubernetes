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
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"io/ioutil"
	"os"
	"testing"

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
		altNames := certutil.AltNames{}
		_, _, actual := newServerKeyAndCert(caCert, caKey, altNames)
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
		_, _, actual := NewClientKeyAndCert(caCert, caKey)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed NewClientKeyAndCert:\n\texpected: %t\n\t  actual: %t",
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
