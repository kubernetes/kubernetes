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
	"errors"
	"sync/atomic"
	"time"

	"k8s.io/apiserver/pkg/storage/value"
	kaes "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
)

const (
	// maxUsage with 2^21 is a very defensive value. 2^32 is more commonly used.
	usage    uint32 = 1 << 21
	maxUsage uint32 = 1 << 31
	// keySize is the key size in bytes
	keySize = 128 / 8
)

var (
	skew = time.Hour * 24
	week = time.Hour * 24 * 7

	// ErrKeyExpired means that the expiration time of a key has come and it shouldn't be used any more.
	ErrKeyExpired = errors.New("key is out of date and shouldn't be used anymore for encryption")
)

// AESGCM is a struct that contains a key of random bytes and additional
// information to re-use it in a safe way.
type AESGCM struct {
	key     []byte
	counter uint32
	expiry  time.Time

	transformer value.Transformer
}

// NewAESGCM returns a pointer to a AESGCM with a key and cipher.AEAD created.
func NewAESGCM(key []byte) (*AESGCM, error) {
	return fromKey(key, 0, time.Now().Add(week))
}

// FromKey initializes a cipher.AEAD from bytes, which is marked as expired.
// It is marked as expired, because it is unknown how often it is already used.
func FromKey(key []byte) (*AESGCM, error) {
	return fromKey(
		key,
		1<<31,
		time.Now().Add(-week).Add(-skew),
	)
}

func fromKey(key []byte, counter uint32, expiry time.Time) (*AESGCM, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	return &AESGCM{
		counter:     counter,
		expiry:      expiry,
		key:         key,
		transformer: kaes.NewGCMTransformer(block),
	}, nil

}

func (c *AESGCM) isReplaceable() bool {
	return c.counter > usage || time.Now().After(c.expiry)
}

// Encrypt encrypts given plaintext. The nonce is prepended. Therefore any
// change to the standard nonceSize is a breaking change.
func (c *AESGCM) Encrypt(ctx context.Context, plaintext []byte) ([]byte, error) {
	if c.counter > maxUsage || time.Now().After(c.expiry) {
		return nil, ErrKeyExpired
	}

	_ = atomic.AddUint32(&c.counter, 1)

	return c.transformer.TransformToStorage(ctx, plaintext, value.DefaultContext{})
}

// Decrypt decrypts a ciphertext. The nonce is assumed to be prepended.
// Therefore any change to the standard nonceSize is a breaking change.
func (k *AESGCM) Decrypt(ctx context.Context, ciphertext []byte) ([]byte, error) {
	plaintext, _, err := k.transformer.TransformFromStorage(ctx, ciphertext, value.DefaultContext{})
	return plaintext, err
}
