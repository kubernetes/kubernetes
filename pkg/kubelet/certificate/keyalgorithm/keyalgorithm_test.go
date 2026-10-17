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

package keyalgorithm

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/mldsa"
	"crypto/rsa"
	"testing"

	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/utils/ptr"
)

func TestKeyGeneratorFunc(t *testing.T) {
	tests := []struct {
		name      string
		algorithm *kubeletconfig.CertificateKeyAlgorithmType
		wantType  string
		validate  func(t *testing.T, key interface{})
		wantErr   bool
	}{
		{
			name:      "nil defaults to ECDSA P-256",
			algorithm: nil,
			wantType:  "*ecdsa.PrivateKey",
			validate: func(t *testing.T, key interface{}) {
				k := key.(*ecdsa.PrivateKey)
				if k.Curve != elliptic.P256() {
					t.Errorf("expected P-256 curve, got %s", k.Curve.Params().Name)
				}
			},
		},
		{
			name:      "ECDSA-P256",
			algorithm: ptr.To(kubeletconfig.CertificateKeyAlgorithmECDSAP256),
			wantType:  "*ecdsa.PrivateKey",
			validate: func(t *testing.T, key interface{}) {
				k := key.(*ecdsa.PrivateKey)
				if k.Curve != elliptic.P256() {
					t.Errorf("expected P-256 curve, got %s", k.Curve.Params().Name)
				}
			},
		},
		{
			name:      "ECDSA-P384",
			algorithm: ptr.To(kubeletconfig.CertificateKeyAlgorithmECDSAP384),
			wantType:  "*ecdsa.PrivateKey",
			validate: func(t *testing.T, key interface{}) {
				k := key.(*ecdsa.PrivateKey)
				if k.Curve != elliptic.P384() {
					t.Errorf("expected P-384 curve, got %s", k.Curve.Params().Name)
				}
			},
		},
		{
			name:      "RSA-2048",
			algorithm: ptr.To(kubeletconfig.CertificateKeyAlgorithmRSA2048),
			wantType:  "*rsa.PrivateKey",
			validate: func(t *testing.T, key interface{}) {
				k := key.(*rsa.PrivateKey)
				if k.N.BitLen() != 2048 {
					t.Errorf("expected 2048-bit key, got %d", k.N.BitLen())
				}
			},
		},
		{
			name:      "RSA-3072",
			algorithm: ptr.To(kubeletconfig.CertificateKeyAlgorithmRSA3072),
			wantType:  "*rsa.PrivateKey",
			validate: func(t *testing.T, key interface{}) {
				k := key.(*rsa.PrivateKey)
				if k.N.BitLen() != 3072 {
					t.Errorf("expected 3072-bit key, got %d", k.N.BitLen())
				}
			},
		},
		{
			name:      "RSA-4096",
			algorithm: ptr.To(kubeletconfig.CertificateKeyAlgorithmRSA4096),
			wantType:  "*rsa.PrivateKey",
			validate: func(t *testing.T, key interface{}) {
				k := key.(*rsa.PrivateKey)
				if k.N.BitLen() != 4096 {
					t.Errorf("expected 4096-bit key, got %d", k.N.BitLen())
				}
			},
		},
		{
			name:      "ML-DSA-44",
			algorithm: ptr.To(kubeletconfig.CertificateKeyAlgorithmMLDSA44),
			wantType:  "*mldsa.PrivateKey",
			validate: func(t *testing.T, key interface{}) {
				k := key.(*mldsa.PrivateKey)
				if k.PublicKey().Parameters() != mldsa.MLDSA44() {
					t.Errorf("expected ML-DSA-44 parameters")
				}
			},
		},
		{
			name:      "ML-DSA-65",
			algorithm: ptr.To(kubeletconfig.CertificateKeyAlgorithmMLDSA65),
			wantType:  "*mldsa.PrivateKey",
			validate: func(t *testing.T, key interface{}) {
				k := key.(*mldsa.PrivateKey)
				if k.PublicKey().Parameters() != mldsa.MLDSA65() {
					t.Errorf("expected ML-DSA-65 parameters")
				}
			},
		},
		{
			name:      "ML-DSA-87",
			algorithm: ptr.To(kubeletconfig.CertificateKeyAlgorithmMLDSA87),
			wantType:  "*mldsa.PrivateKey",
			validate: func(t *testing.T, key interface{}) {
				k := key.(*mldsa.PrivateKey)
				if k.PublicKey().Parameters() != mldsa.MLDSA87() {
					t.Errorf("expected ML-DSA-87 parameters")
				}
			},
		},
		{
			name:      "unsupported algorithm",
			algorithm: ptr.To(kubeletconfig.CertificateKeyAlgorithmType("BOGUS-999")),
			wantErr:   true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			genFunc := KeyGeneratorFunc(tc.algorithm)
			key, err := genFunc()
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if key == nil {
				t.Fatal("expected non-nil key")
			}
			if key.Public() == nil {
				t.Fatal("expected non-nil public key")
			}
			tc.validate(t, key)
		})
	}
}
