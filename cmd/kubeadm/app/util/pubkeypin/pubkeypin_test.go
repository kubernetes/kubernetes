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

package pubkeypin

import (
	"crypto/x509"
	_ "embed"
	"encoding/pem"
	"strings"
	"testing"
)

// testCertPEM is a simple self-signed test certificate issued with the openssl CLI:
// openssl req -new -newkey rsa:2048 -days 36500 -nodes -x509 -keyout /dev/null -out test.crt
//
//go:embed testdata/test-cert.pem
var testCertPEM string

// expectedHash can be verified using the openssl CLI.
const expectedHash = `sha256:345959acb2c3b2feb87d281961c893f62a314207ef02599f1cc4a5fb255480b3`

// testCert2PEM is a second test cert generated the same way as testCertPEM
//
//go:embed testdata/test-cert2.pem
var testCert2PEM string

// testCert is a small helper to get a test x509.Certificate from the PEM constants
func testCert(t *testing.T, pemString string) *x509.Certificate {
	// Decode the example certificate from a PEM file into a PEM block
	pemBlock, _ := pem.Decode([]byte(pemString))
	if pemBlock == nil {
		t.Fatal("failed to parse test certificate PEM")
		return nil
	}

	// Parse the PEM block into an x509.Certificate
	result, err := x509.ParseCertificate(pemBlock.Bytes)
	if err != nil {
		t.Fatalf("failed to parse test certificate: %v", err)
		return nil
	}
	return result
}

func TestSet(t *testing.T) {
	s := NewSet()
	if !s.Empty() {
		t.Error("expected a new set to be empty")
		return
	}
	err := s.Allow("xyz")
	if err == nil || !s.Empty() {
		t.Error("expected allowing junk to fail")
		return
	}

	err = s.Allow("0011223344")
	if err == nil || !s.Empty() {
		t.Error("expected allowing something too short to fail")
		return
	}

	err = s.Allow(expectedHash + expectedHash)
	if err == nil || !s.Empty() {
		t.Error("expected allowing something too long to fail")
		return
	}

	err = s.CheckAny([]*x509.Certificate{testCert(t, testCertPEM)})
	if err == nil {
		t.Error("expected test cert to not be allowed (yet)")
		return
	}

	err = s.Allow(strings.ToUpper(expectedHash))
	if err != nil || s.Empty() {
		t.Error("expected allowing uppercase expectedHash to succeed")
		return
	}

	err = s.CheckAny([]*x509.Certificate{testCert(t, testCertPEM)})
	if err != nil {
		t.Errorf("expected test cert to be allowed, but got back: %v", err)
		return
	}

	err = s.CheckAny([]*x509.Certificate{testCert(t, testCert2PEM)})
	if err == nil {
		t.Error("expected the second test cert to be disallowed")
		return
	}

	s = NewSet() // keep set empty
	hashes := []string{
		`sha256:0000000000000000000000000000000000000000000000000000000000000000`,
		`sha256:0000000000000000000000000000000000000000000000000000000000000001`,
	}
	err = s.Allow(hashes...)
	if err != nil || len(s.sha256Hashes) != 2 {
		t.Error("expected allowing multiple hashes to succeed")
		return
	}
}

func TestHash(t *testing.T) {
	actualHash := Hash(testCert(t, testCertPEM))
	if actualHash != expectedHash {
		t.Errorf(
			"failed to Hash() to the expected value\n\texpected: %q\n\t  actual: %q",
			expectedHash,
			actualHash,
		)
	}
}
