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
	"crypto/cipher"
	"fmt"
)

const (
	// TagSize is the tag size in bytes for AES-128-GCM-SHA256,
	// AES-256-GCM-SHA384, and CHACHA20-POLY1305-SHA256.
	TagSize = 16
	// NonceSize is the size of the nonce in number of bytes for
	// AES-128-GCM-SHA256, AES-256-GCM-SHA384, and CHACHA20-POLY1305-SHA256.
	NonceSize = 12
	// SHA256DigestSize is the digest size of sha256 in bytes.
	SHA256DigestSize = 32
	// SHA384DigestSize is the digest size of sha384 in bytes.
	SHA384DigestSize = 48
)

// sliceForAppend takes a slice and a requested number of bytes. It returns a
// slice with the contents of the given slice followed by that many bytes and a
// second slice that aliases into it and contains only the extra bytes. If the
// original slice has sufficient capacity then no allocation is performed.
func sliceForAppend(in []byte, n int) (head, tail []byte) {
	if total := len(in) + n; cap(in) >= total {
		head = in[:total]
	} else {
		head = make([]byte, total)
		copy(head, in)
	}
	tail = head[len(in):]
	return head, tail
}

// encrypt is the encryption function for an AEAD crypter. aead determines
// the type of AEAD crypter. dst can contain bytes at the beginning of the
// ciphertext that will not be encrypted but will be authenticated. If dst has
// enough capacity to hold these bytes, the ciphertext and the tag, no
// allocation and copy operations will be performed. dst and plaintext may
// fully overlap or not at all.
func encrypt(aead cipher.AEAD, dst, plaintext, nonce, aad []byte) ([]byte, error) {
	if len(nonce) != NonceSize {
		return nil, fmt.Errorf("nonce size must be %d bytes. received: %d", NonceSize, len(nonce))
	}
	// If we need to allocate an output buffer, we want to include space for
	// the tag to avoid forcing the caller to reallocate as well.
	dlen := len(dst)
	dst, out := sliceForAppend(dst, len(plaintext)+TagSize)
	data := out[:len(plaintext)]
	copy(data, plaintext) // data may fully overlap plaintext

	// Seal appends the ciphertext and the tag to its first argument and
	// returns the updated slice. However, sliceForAppend above ensures that
	// dst has enough capacity to avoid a reallocation and copy due to the
	// append.
	dst = aead.Seal(dst[:dlen], nonce, data, aad)
	return dst, nil
}

// decrypt is the decryption function for an AEAD crypter, where aead determines
// the type of AEAD crypter, and dst the destination bytes for the decrypted
// ciphertext. The dst buffer may fully overlap with plaintext or not at all.
func decrypt(aead cipher.AEAD, dst, ciphertext, nonce, aad []byte) ([]byte, error) {
	if len(nonce) != NonceSize {
		return nil, fmt.Errorf("nonce size must be %d bytes. received: %d", NonceSize, len(nonce))
	}
	// If dst is equal to ciphertext[:0], ciphertext storage is reused.
	plaintext, err := aead.Open(dst, nonce, ciphertext, aad)
	if err != nil {
		return nil, fmt.Errorf("message auth failed: %v", err)
	}
	return plaintext, nil
}
