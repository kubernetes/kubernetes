/*
Copyright The Kubernetes Authors.

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

package utils_test

import (
	"crypto"
	"crypto/x509"
	"testing"

	certutil "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/keyutil"
	"k8s.io/kubernetes/test/utils"
)

func TestNewSignedCertKeyUsage(t *testing.T) {
	const ecdsaPrivateKey = `-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIP6Qw6dHDiLsSnLXUhQVTPE0fTQQrj3XSbiQAZPXnk5+oAoGCCqGSM49
AwEHoUQDQgAEZZzi1u5f2/AEGFI/HYUhU+u6cTK1q2bbtE7r1JMK+/sQA5sNAp+7
Vdc3psr1OaNzyTyuhTECyRdFKXm63cMnGg==
-----END EC PRIVATE KEY-----`

	rsaKey, err := keyutil.ParsePrivateKeyPEM(utils.LocalhostKey)
	if err != nil {
		t.Fatalf("Failed to load RSA key fixture: %v", err)
	}
	ecdsaKey, err := keyutil.ParsePrivateKeyPEM([]byte(ecdsaPrivateKey))
	if err != nil {
		t.Fatalf("Failed to load ECDSA key fixture: %v", err)
	}

	tests := []struct {
		name             string
		key              crypto.Signer
		caKey            crypto.Signer
		expectedKeyUsage x509.KeyUsage
	}{
		{
			name:             "RSA-signed-by-ECDSA",
			key:              rsaKey.(crypto.Signer),
			caKey:            ecdsaKey.(crypto.Signer),
			expectedKeyUsage: x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		},
		{
			name:             "ECDSA-signed-by-RSA",
			key:              ecdsaKey.(crypto.Signer),
			caKey:            rsaKey.(crypto.Signer),
			expectedKeyUsage: x509.KeyUsageDigitalSignature,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			caCert, err := certutil.NewSelfSignedCACert(certutil.Config{CommonName: "test-ca"}, tt.caKey)
			if err != nil {
				t.Fatalf("Failed to create CA certificate: %v", err)
			}
			cert, err := utils.NewSignedCert(&certutil.Config{
				CommonName: "test-client",
				Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			}, tt.key, caCert, tt.caKey)
			if err != nil {
				t.Fatalf("Failed to create signed certificate: %v", err)
			}
			if cert.KeyUsage != tt.expectedKeyUsage {
				t.Errorf("expected key usage %v, got %v", tt.expectedKeyUsage, cert.KeyUsage)
			}
		})
	}
}
