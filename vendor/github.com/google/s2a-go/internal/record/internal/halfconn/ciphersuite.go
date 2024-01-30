/*
 *
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package halfconn

import (
	"crypto/sha256"
	"crypto/sha512"
	"fmt"
	"hash"

	s2apb "github.com/google/s2a-go/internal/proto/common_go_proto"
	"github.com/google/s2a-go/internal/record/internal/aeadcrypter"
)

// ciphersuite is the interface for retrieving ciphersuite-specific information
// and utilities.
type ciphersuite interface {
	// keySize returns the key size in bytes. This refers to the key used by
	// the AEAD crypter. This is derived by calling HKDF expand on the traffic
	// secret.
	keySize() int
	// nonceSize returns the nonce size in bytes.
	nonceSize() int
	// trafficSecretSize returns the traffic secret size in bytes. This refers
	// to the secret used to derive the traffic key and nonce, as specified in
	// https://tools.ietf.org/html/rfc8446#section-7.
	trafficSecretSize() int
	// hashFunction returns the hash function for the ciphersuite.
	hashFunction() func() hash.Hash
	// aeadCrypter takes a key and creates an AEAD crypter for the ciphersuite
	// using that key.
	aeadCrypter(key []byte) (aeadcrypter.S2AAEADCrypter, error)
}

func newCiphersuite(ciphersuite s2apb.Ciphersuite) (ciphersuite, error) {
	switch ciphersuite {
	case s2apb.Ciphersuite_AES_128_GCM_SHA256:
		return &aesgcm128sha256{}, nil
	case s2apb.Ciphersuite_AES_256_GCM_SHA384:
		return &aesgcm256sha384{}, nil
	case s2apb.Ciphersuite_CHACHA20_POLY1305_SHA256:
		return &chachapolysha256{}, nil
	default:
		return nil, fmt.Errorf("unrecognized ciphersuite: %v", ciphersuite)
	}
}

// aesgcm128sha256 is the AES-128-GCM-SHA256 implementation of the ciphersuite
// interface.
type aesgcm128sha256 struct{}

func (aesgcm128sha256) keySize() int                   { return aeadcrypter.AES128GCMKeySize }
func (aesgcm128sha256) nonceSize() int                 { return aeadcrypter.NonceSize }
func (aesgcm128sha256) trafficSecretSize() int         { return aeadcrypter.SHA256DigestSize }
func (aesgcm128sha256) hashFunction() func() hash.Hash { return sha256.New }
func (aesgcm128sha256) aeadCrypter(key []byte) (aeadcrypter.S2AAEADCrypter, error) {
	return aeadcrypter.NewAESGCM(key)
}

// aesgcm256sha384 is the AES-256-GCM-SHA384 implementation of the ciphersuite
// interface.
type aesgcm256sha384 struct{}

func (aesgcm256sha384) keySize() int                   { return aeadcrypter.AES256GCMKeySize }
func (aesgcm256sha384) nonceSize() int                 { return aeadcrypter.NonceSize }
func (aesgcm256sha384) trafficSecretSize() int         { return aeadcrypter.SHA384DigestSize }
func (aesgcm256sha384) hashFunction() func() hash.Hash { return sha512.New384 }
func (aesgcm256sha384) aeadCrypter(key []byte) (aeadcrypter.S2AAEADCrypter, error) {
	return aeadcrypter.NewAESGCM(key)
}

// chachapolysha256 is the ChaChaPoly-SHA256 implementation of the ciphersuite
// interface.
type chachapolysha256 struct{}

func (chachapolysha256) keySize() int                   { return aeadcrypter.Chacha20Poly1305KeySize }
func (chachapolysha256) nonceSize() int                 { return aeadcrypter.NonceSize }
func (chachapolysha256) trafficSecretSize() int         { return aeadcrypter.SHA256DigestSize }
func (chachapolysha256) hashFunction() func() hash.Hash { return sha256.New }
func (chachapolysha256) aeadCrypter(key []byte) (aeadcrypter.S2AAEADCrypter, error) {
	return aeadcrypter.NewChachaPoly(key)
}
