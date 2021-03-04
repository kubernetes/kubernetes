/*
Copyright 2019 The Kubernetes Authors.

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

package crypto

import (
	"github.com/pkg/errors"

	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
)

// CreateRandBytes returns a cryptographically secure slice of random bytes with a given size
func CreateRandBytes(size uint32) ([]byte, error) {
	bytes := make([]byte, size)
	if _, err := rand.Read(bytes); err != nil {
		return nil, err
	}
	return bytes, nil
}

// EncryptBytes takes a byte slice of raw data and an encryption key and returns an encrypted byte slice of data.
// The key must be an AES key, either 16, 24, or 32 bytes to select AES-128, AES-192, or AES-256
func EncryptBytes(data, key []byte) ([]byte, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	nonce, err := CreateRandBytes(uint32(gcm.NonceSize()))
	if err != nil {
		return nil, err
	}
	return gcm.Seal(nonce, nonce, data, nil), nil
}

// DecryptBytes takes a byte slice of encrypted data and an encryption key and returns a decrypted byte slice of data.
// The key must be an AES key, either 16, 24, or 32 bytes to select AES-128, AES-192, or AES-256
func DecryptBytes(data, key []byte) ([]byte, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	nonceSize := gcm.NonceSize()
	if len(data) < nonceSize {
		return nil, errors.New("size of data is less than the nonce")
	}

	nonce, out := data[:nonceSize], data[nonceSize:]
	out, err = gcm.Open(nil, nonce, out, nil)
	if err != nil {
		return nil, err
	}
	return out, nil
}
