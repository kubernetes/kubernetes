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
	"crypto"
	cryptorand "crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"testing"

	"k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/keyutil"
)

const COMMON_NAME = "foo.example.com"

const (
	// rsaPrivateKey is an RSA-2048 private key in PKCS#1 format
	// openssl genrsa -traditional -out rsa2048.pem 2048
	rsaPrivateKey = `-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA92mVjhBKOFsdxFzb/Pjq+7b5TJlODAdY5hK+WxLZTIrfhDPq
FWrGKdjSNiHbXrdEtwJh9V+RqPZVSN3aWy1224RgkyNdMJsXhJKuCC24ZKY8SXtW
xuTYmMRaMnCsv6QBGRTIbZ2EFbAObVM7lDyv1VqY3amZIWFQMlZ9CNpxDSPa5yi4
3gopbXkne0oGNmey9X0qtpk7NMZIgAL6Zz4rZ30bcfC2ag6RLOFI2E/c4n8c38R8
9MfXfLkj8/Cxo4JfI9NvRCpPOpFO8d/ZtWVUuIrBQN+Y7tkN2T60Qq/TkKXUrhDe
fwlTlktZVJ/GztLYU41b2GcWsh/XO+PH831rmwIDAQABAoIBAQCC9c6GDjVbM0/E
WurPMusfJjE7zII1d8YkspM0HfwLug6qKdikUYpnKC/NG4rEzfl/bbFwco/lgc6O
7W/hh2U8uQttlvCDA/Uk5YddKOZL0Hpk4vaB/SxxYK3luSKXpjY2knutGg2KdVCN
qdsFkkH4iyYTXuyBcMNEgedZQldI/kEujIH/L7FE+DF5TMzT4lHhozDoG+fy564q
qVGUZXJn0ubc3GaPn2QOLNNM44sfYA4UJCpKBXPu85bvNObjxVQO4WqwwxU1vRnL
UUsaGaelhSVJCo0dVPRvrfPPKZ09HTwpy40EkgQo6VriFc1EBoQDjENLbAJv9OfQ
aCc9wiZhAoGBAP/8oEy48Zbb0P8Vdy4djf5tfBW8yXFLWzXewJ4l3itKS1r42nbX
9q3cJsgRTQm8uRcMIpWxsc3n6zG+lREvTkoTB3ViI7+uQPiqA+BtWyNy7jzufFke
ONKZfg7QxxmYRWZBRnoNGNbMpNeERuLmhvQuom9D1WbhzAYJbfs/O4WTAoGBAPds
2FNDU0gaesFDdkIUGq1nIJqRQDW485LXZm4pFqBFxdOpbdWRuYT2XZjd3fD0XY98
Nhkpb7NTMCuK3BdKcqIptt+cK+quQgYid0hhhgZbpCQ5AL6c6KgyjgpYlh2enzU9
Zo3yg8ej1zbbA11sBlhX+5iO2P1u5DG+JHLwUUbZAoGAUwaU102EzfEtsA4+QW7E
hyjrfgFlNKHES4yb3K9bh57pIfBkqvcQwwMMcQdrfSUAw0DkVrjzel0mI1Q09QXq
1ould6UFAz55RC2gZEITtUOpkYmoOx9aPrQZ9qQwb1S77ZZuTVfCHqjxLhVxCFbM
npYhiQTvShciHTMhwMOZgpECgYAVV5EtVXBYltgh1YTc3EkUzgF087R7LdHsx6Gx
POATwRD4WfP8aQ58lpeqOPEM+LcdSlSMRRO6fyF3kAm+BJDwxfJdRWZQXumZB94M
I0VhRQRaj4Qt7PDwmTPBVrTUJzuKZxpyggm17b8Bn1Ch/VBqzGQKW8AB1E/grosM
UwhfuQKBgQC2JO/iqTQScHClf0qlItCJsBuVukFmSAVCkpOD8YdbdlPdOOwSk1wQ
C0eAlsC3BCMvkpidKQmra6IqIrvTGI6EFgkrb3aknWdup2w8j2udYCNqyE3W+fVe
p8FdYQ1FkACQ+daO5VlClL/9l0sGjKXlNKbpmJ2H4ngZmXj5uGmxuQ==
-----END RSA PRIVATE KEY-----`

	// ecdsaPrivateKey is an ECDSA-P256 private key in SEC 1 format
	// openssl ecparam -name prime256v1 -genkey -noout -out ecdsa256.pem
	ecdsaPrivateKey = `-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIP6Qw6dHDiLsSnLXUhQVTPE0fTQQrj3XSbiQAZPXnk5+oAoGCCqGSM49
AwEHoUQDQgAEZZzi1u5f2/AEGFI/HYUhU+u6cTK1q2bbtE7r1JMK+/sQA5sNAp+7
Vdc3psr1OaNzyTyuhTECyRdFKXm63cMnGg==
-----END EC PRIVATE KEY-----`

	// ed25519PrivateKey is an Ed25519 private key in PKCS#8 format
	// openssl genpkey -algorithm ED25519
	ed25519PrivateKey = `-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEIFAoQzL1vNqV2kXNiTxRyqIv924/F9BgOIHuqX8RV4YO
-----END PRIVATE KEY-----`

	// mldsaPrivateKey is an ML-DSA-65 private key in PKCS#8 format
	// openssl genpkey -algorithm ML-DSA-65 -out mldsa65.pem
	mldsaPrivateKey = `-----BEGIN PRIVATE KEY-----
MDQCAQAwCwYJYIZIAWUDBAMSBCKAINgd6EB5JBi28ztvyFxNM3YCNcldR8MmSO04
zfKOIXsK
-----END PRIVATE KEY-----`
)

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

func TestNewSelfSignedCACertKeyUsage(t *testing.T) {
	const caKeyUsage = x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign
	tests := []struct {
		name             string
		getKey           func() (crypto.Signer, error)
		expectedKeyUsage x509.KeyUsage
	}{
		{
			name: "RSA",
			getKey: func() (crypto.Signer, error) {
				key, err := keyutil.ParsePrivateKeyPEM([]byte(rsaPrivateKey))
				if err != nil {
					return nil, err
				}
				return key.(crypto.Signer), nil
			},
			expectedKeyUsage: caKeyUsage | x509.KeyUsageKeyEncipherment,
		},
		{
			name: "ECDSA-P256",
			getKey: func() (crypto.Signer, error) {
				key, err := keyutil.ParsePrivateKeyPEM([]byte(ecdsaPrivateKey))
				if err != nil {
					return nil, err
				}
				return key.(crypto.Signer), nil
			},
			expectedKeyUsage: caKeyUsage,
		},
		{
			name: "Ed25519",
			getKey: func() (crypto.Signer, error) {
				key, err := keyutil.ParsePrivateKeyPEM([]byte(ed25519PrivateKey))
				if err != nil {
					return nil, err
				}
				return key.(crypto.Signer), nil
			},
			expectedKeyUsage: caKeyUsage,
		},
		{
			name: "ML-DSA-65",
			getKey: func() (crypto.Signer, error) {
				key, err := keyutil.ParsePrivateKeyPEM([]byte(mldsaPrivateKey))
				if err != nil {
					return nil, err
				}
				return key.(crypto.Signer), nil
			},
			expectedKeyUsage: caKeyUsage,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			key, err := tt.getKey()
			if err != nil {
				t.Fatalf("Failed to load key fixture: %v", err)
			}
			caCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: COMMON_NAME}, key)
			if err != nil {
				t.Fatalf("Failed to create CA certificate: %v", err)
			}
			if caCert.KeyUsage != tt.expectedKeyUsage {
				t.Errorf("expected key usage %v, got %v", tt.expectedKeyUsage, caCert.KeyUsage)
			}
		})
	}
}
