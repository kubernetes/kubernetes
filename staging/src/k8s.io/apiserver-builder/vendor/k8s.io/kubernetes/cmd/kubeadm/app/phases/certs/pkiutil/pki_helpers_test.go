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

func TestWriteCert(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	caCert := &x509.Certificate{}
	actual := WriteCert(tmpdir, "foo", caCert)
	if actual != nil {
		t.Errorf(
			"failed WriteCertAndKey with an error: %v",
			actual,
		)
	}
}

func TestWriteKey(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	caKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("Couldn't create rsa Private Key")
	}
	actual := WriteKey(tmpdir, "foo", caKey)
	if actual != nil {
		t.Errorf(
			"failed WriteCertAndKey with an error: %v",
			actual,
		)
	}
}

func TestWritePublicKey(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	caKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("Couldn't create rsa Private Key")
	}
	actual := WritePublicKey(tmpdir, "foo", &caKey.PublicKey)
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

func TestTryLoadCertFromDisk(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	caCert, _, err := NewCertificateAuthority()
	if err != nil {
		t.Errorf(
			"failed to create cert and key with an error: %v",
			err,
		)
	}
	err = WriteCert(tmpdir, "foo", caCert)
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
		_, actual := TryLoadCertFromDisk(rt.path, rt.name)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed TryLoadCertAndKeyFromDisk:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
		}
	}
}

func TestTryLoadKeyFromDisk(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	_, caKey, err := NewCertificateAuthority()
	if err != nil {
		t.Errorf(
			"failed to create cert and key with an error: %v",
			err,
		)
	}
	err = WriteKey(tmpdir, "foo", caKey)
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
		_, actual := TryLoadKeyFromDisk(rt.path, rt.name)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed TryLoadCertAndKeyFromDisk:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
		}
	}
}

func TestPathsForCertAndKey(t *testing.T) {
	crtPath, keyPath := pathsForCertAndKey("/foo", "bar")
	if crtPath != "/foo/bar.crt" {
		t.Errorf("unexpected certificate path: %s", crtPath)
	}
	if keyPath != "/foo/bar.key" {
		t.Errorf("unexpected key path: %s", keyPath)
	}
}

func TestPathForCert(t *testing.T) {
	crtPath := pathForCert("/foo", "bar")
	if crtPath != "/foo/bar.crt" {
		t.Errorf("unexpected certificate path: %s", crtPath)
	}
}

func TestPathForKey(t *testing.T) {
	keyPath := pathForKey("/foo", "bar")
	if keyPath != "/foo/bar.key" {
		t.Errorf("unexpected certificate path: %s", keyPath)
	}
}

func TestPathForPublicKey(t *testing.T) {
	pubPath := pathForPublicKey("/foo", "bar")
	if pubPath != "/foo/bar.pub" {
		t.Errorf("unexpected certificate path: %s", pubPath)
	}
}
