/*
Copyright 2022 The Kubernetes Authors.

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

package encryption

import (
	"context"
	"crypto/aes"
	"crypto/rand"
	"errors"

	"k8s.io/apiserver/pkg/storage/value"
	kaes "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
)

const (
	// keySize is the key size in bytes
	keySize = 128 / 8
)

var (
	ErrEmptyPlaintext = errors.New("plaintext for encryption is empty")
)

// AESGCM is a struct that contains a key of random bytes and additional
// information to re-use it in a safe way.
type AESGCM struct {
	key []byte

	aesGCM value.Transformer
}

// NewAESGCM returns a pointer to a AESGCM with a key and cipher.AEAD created.
func NewAESGCM() (*AESGCM, error) {
	key, err := randomBytes(keySize)
	if err != nil {
		return nil, err
	}

	return FromKey(key)
}

// FromKey initializes a cipher.AEAD from bytes so the block cipher doesn't need
// to be initialized every time.
func FromKey(key []byte) (*AESGCM, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	transformer, _, err := kaes.NewGCMTransformer(block)
	if err != nil {
		return nil, err
	}

	return &AESGCM{
		key:    key,
		aesGCM: transformer,
	}, nil

}

// Key returns the internal Key used for encryption. One should be cautious with
// this value as it is confidential data that must not leak.
func (c *AESGCM) Key() []byte {
	return c.key
}

// Encrypt encrypts given plaintext. The nonce is prepended. Therefore any
// change to the standard nonceSize is a breaking change.
func (c *AESGCM) Encrypt(ctx context.Context, plaintext []byte) ([]byte, error) {
	return c.aesGCM.TransformToStorage(ctx, plaintext, value.DefaultContext)
}

// Decrypt decrypts a ciphertext. The nonce is assumed to be prepended.
// Therefore any change to the standard nonceSize is a breaking change.
func (k *AESGCM) Decrypt(ctx context.Context, ciphertext []byte) ([]byte, error) {
	plaintext, _, err := k.aesGCM.TransformFromStorage(ctx, ciphertext, value.DefaultContext)
	return plaintext, err
}

// randomBytes generates length amount of bytes.
func randomBytes(length int) (key []byte, err error) {
	key = make([]byte, length)

	// TODO@ibihim: diff between rand.Read and ioutil.ReadFull?
	if _, err = rand.Read(key); err != nil {
		return nil, err
	}

	return key, nil
}
