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

// Package aeadcrypter provides the interface for AEAD cipher implementations
// used by S2A's record protocol.
package aeadcrypter

// S2AAEADCrypter is the interface for an AEAD cipher used by the S2A record
// protocol.
type S2AAEADCrypter interface {
	// Encrypt encrypts the plaintext and computes the tag of dst and plaintext.
	// dst and plaintext may fully overlap or not at all.
	Encrypt(dst, plaintext, nonce, aad []byte) ([]byte, error)
	// Decrypt decrypts ciphertext and verifies the tag. dst and ciphertext may
	// fully overlap or not at all.
	Decrypt(dst, ciphertext, nonce, aad []byte) ([]byte, error)
	// TagSize returns the tag size in bytes.
	TagSize() int
}
