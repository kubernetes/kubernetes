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

package aeadcrypter

import (
	"crypto/aes"
	"crypto/cipher"
	"fmt"
)

// Supported key sizes in bytes.
const (
	AES128GCMKeySize = 16
	AES256GCMKeySize = 32
)

// aesgcm is the struct that holds an AES-GCM cipher for the S2A AEAD crypter.
type aesgcm struct {
	aead cipher.AEAD
}

// NewAESGCM creates an AES-GCM crypter instance. Note that the key must be
// either 128 bits or 256 bits.
func NewAESGCM(key []byte) (S2AAEADCrypter, error) {
	if len(key) != AES128GCMKeySize && len(key) != AES256GCMKeySize {
		return nil, fmt.Errorf("%d or %d bytes, given: %d", AES128GCMKeySize, AES256GCMKeySize, len(key))
	}
	c, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	a, err := cipher.NewGCM(c)
	if err != nil {
		return nil, err
	}
	return &aesgcm{aead: a}, nil
}

// Encrypt is the encryption function. dst can contain bytes at the beginning of
// the ciphertext that will not be encrypted but will be authenticated. If dst
// has enough capacity to hold these bytes, the ciphertext and the tag, no
// allocation and copy operations will be performed. dst and plaintext may
// fully overlap or not at all.
func (s *aesgcm) Encrypt(dst, plaintext, nonce, aad []byte) ([]byte, error) {
	return encrypt(s.aead, dst, plaintext, nonce, aad)
}

func (s *aesgcm) Decrypt(dst, ciphertext, nonce, aad []byte) ([]byte, error) {
	return decrypt(s.aead, dst, ciphertext, nonce, aad)
}

func (s *aesgcm) TagSize() int {
	return TagSize
}
