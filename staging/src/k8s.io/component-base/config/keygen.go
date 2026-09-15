/*
Copyright 2025 The Kubernetes Authors.

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

package config

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/mldsa"
	cryptorand "crypto/rand"
	"crypto/rsa"
	"fmt"
)

// KeyGeneratorFunc returns a GenerateKey function for the specified algorithm.
func KeyGeneratorFunc(algorithm string) func() (crypto.Signer, error) {
	return func() (crypto.Signer, error) {
		switch EncryptionAlgorithmType(algorithm) {
		case "", EncryptionAlgorithmECDSAP256:
			return ecdsa.GenerateKey(elliptic.P256(), cryptorand.Reader)
		case EncryptionAlgorithmECDSAP384:
			return ecdsa.GenerateKey(elliptic.P384(), cryptorand.Reader)
		case EncryptionAlgorithmRSA2048:
			return rsa.GenerateKey(cryptorand.Reader, 2048)
		case EncryptionAlgorithmRSA3072:
			return rsa.GenerateKey(cryptorand.Reader, 3072)
		case EncryptionAlgorithmRSA4096:
			return rsa.GenerateKey(cryptorand.Reader, 4096)
		case EncryptionAlgorithmMLDSA44:
			return mldsa.GenerateKey(mldsa.MLDSA44())
		case EncryptionAlgorithmMLDSA65:
			return mldsa.GenerateKey(mldsa.MLDSA65())
		case EncryptionAlgorithmMLDSA87:
			return mldsa.GenerateKey(mldsa.MLDSA87())
		default:
			return nil, fmt.Errorf("unsupported key algorithm: %s", algorithm)
		}
	}
}
