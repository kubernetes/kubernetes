/*
Copyright 2021 The Kubernetes Authors.

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

package cert_test

import (
	cryptorand "crypto/rand"
	"crypto/rsa"
	"testing"

	"k8s.io/client-go/util/cert"
)

const COMMON_NAME = "foo.example.com"

// TestSelfSignedCertHasSAN verifies the existing of
// a SAN on the generated self-signed certificate.
// a SAN ensures that the certificate is considered
// valid by default in go 1.15 and above, which
// turns off fallback to Common Name by default.
func TestSelfSignedCertHasSAN(t *testing.T) {
	key, err := rsa.GenerateKey(cryptorand.Reader, 2048)
	if err != nil {
		t.Fatalf("rsa key failed to generate: %v", err)
	}
	selfSignedCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: COMMON_NAME}, key)
	if err != nil {
		t.Fatalf("self signed certificate failed to generate: %v", err)
	}
	if len(selfSignedCert.DNSNames) == 0 {
		t.Fatalf("self signed certificate has zero DNS names.")
	}
}
