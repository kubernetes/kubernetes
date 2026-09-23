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
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/mldsa"
	cryptorand "crypto/rand"
	"crypto/rsa"
	"fmt"

	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

// KeyGeneratorFunc returns a GenerateKey function for the specified algorithm.
// If algorithm is nil, defaults to ECDSA P-256.
func KeyGeneratorFunc(algorithm *kubeletconfig.CertificateKeyAlgorithmType) func() (crypto.Signer, error) {
	return func() (crypto.Signer, error) {
		var algo kubeletconfig.CertificateKeyAlgorithmType
		if algorithm != nil {
			algo = *algorithm
		}
		switch algo {
		case "", kubeletconfig.CertificateKeyAlgorithmECDSAP256:
			return ecdsa.GenerateKey(elliptic.P256(), cryptorand.Reader)
		case kubeletconfig.CertificateKeyAlgorithmECDSAP384:
			return ecdsa.GenerateKey(elliptic.P384(), cryptorand.Reader)
		case kubeletconfig.CertificateKeyAlgorithmRSA2048:
			return rsa.GenerateKey(cryptorand.Reader, 2048)
		case kubeletconfig.CertificateKeyAlgorithmRSA3072:
			return rsa.GenerateKey(cryptorand.Reader, 3072)
		case kubeletconfig.CertificateKeyAlgorithmRSA4096:
			return rsa.GenerateKey(cryptorand.Reader, 4096)
		case kubeletconfig.CertificateKeyAlgorithmMLDSA44:
			return mldsa.GenerateKey(mldsa.MLDSA44())
		case kubeletconfig.CertificateKeyAlgorithmMLDSA65:
			return mldsa.GenerateKey(mldsa.MLDSA65())
		case kubeletconfig.CertificateKeyAlgorithmMLDSA87:
			return mldsa.GenerateKey(mldsa.MLDSA87())
		default:
			return nil, fmt.Errorf("unsupported key algorithm: %s", algo)
		}
	}
}

// IsMLDSA reports whether algorithm is an ML-DSA variant. TLS only defines the
// ML-DSA signature schemes for TLS 1.3.
func IsMLDSA(algorithm *kubeletconfig.CertificateKeyAlgorithmType) bool {
	if algorithm == nil {
		return false
	}
	switch *algorithm {
	case kubeletconfig.CertificateKeyAlgorithmMLDSA44,
		kubeletconfig.CertificateKeyAlgorithmMLDSA65,
		kubeletconfig.CertificateKeyAlgorithmMLDSA87:
		return true
	}
	return false
}
