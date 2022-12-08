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
	"sync/atomic"
	"time"
)

const (
	// safeUsage with 2^21 is a very defensive value.  2^21 is calculated by
	// assuming a probability of 1/2^80 for the birthday problem.
	safeUsage uint32 = 1 << 21
	// maxUsage is closer to the recommended maximum usage of 2^32 times.
	maxUsage uint32 = 1 << 31
	// spaceUsage is a defensive maximum amount of bytes of data that should be
	// encrypted with a key and a nonce. It is approximately 32 GiByte.
	// The maximum usage is 2^36 - 256 byte, which is approximately 64 GiByte.
	spaceUsage uint64 = 1 << 35

	// keySize is the key size in bytes
	keySize = 128 / 8
)

var (
	// week was picked by during discussions of KMSv2 development.
	week = time.Hour * 24 * 7

	// ErrKeyExpired means that the expiration time of a key has come. It shouldn't be used any more.
	ErrKeyExpired = errors.New("key is out of date and shouldn't be used anymore for encryption")
	// ErrDataTooBig means that the given plaintext is too big. The plaintext needs to be split up.
	ErrDataTooBig = errors.New("plaintext exceeds 32 GiB")
)

// AESGCM is a struct that contains a key of random bytes and additional
// information to re-use it in a safe way.
// AES-GCM has some safety limits until it reaches cryptographic wearout.
// The maximum length of bytes that can be encrypted with one key and nonce is
// ((2^36)-256) bytes.
// The amount of times a key with different random nonces can be used, until the
// probability of re-using a nonce becomes to high is 2^32.
type AESGCM struct {
	key     []byte
	counter uint32
	expiry  time.Time

	transformer Transformer
}

// NewAESGCM returns a pointer to a AESGCM with a key and cipher.AEAD created.
func NewAESGCM() (*AESGCM, error) {
	key, err := RandomBytes(keySize)
	if err != nil {
		return nil, err
	}

	return newAESGCM(key, 0, time.Now().Add(week))
}

func newAESGCM(key []byte, counter uint32, expiry time.Time) (*AESGCM, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	return &AESGCM{
		counter:     counter,
		expiry:      expiry,
		key:         key,
		transformer: NewGCMTransformer(block),
	}, nil
}

// isValid checks if the key is safe to use, by checking the counter and expiry.
func (c *AESGCM) isValid() bool {
	return c.counter < maxUsage && time.Now().Before(c.expiry)
}

// Encrypt encrypts given plaintext. It will fail, if the key is invalid.
func (c *AESGCM) TransformToStorage(ctx context.Context, plaintext []byte, dataCtx Context) ([]byte, error) {
	if uint64(len(plaintext)) >= spaceUsage {
		return nil, ErrDataTooBig
	}

	if !c.isValid() {
		return nil, ErrKeyExpired
	}

	_ = atomic.AddUint32(&c.counter, 1)

	return c.transformer.TransformToStorage(ctx, plaintext, dataCtx)
}

// Decrypt decrypts a ciphertext. The nonce is assumed to be prepended.
// Therefore any change to the standard nonceSize is a breaking change.
func (k *AESGCM) TransformFromStorage(ctx context.Context, ciphertext []byte, dataCtx Context) ([]byte, bool, error) {
	plaintext, stale, err := k.transformer.TransformFromStorage(ctx, ciphertext, dataCtx)
	return plaintext, stale, err
}

// RandomBytes generates length amount of bytes.
func RandomBytes(length int) (key []byte, err error) {
	key = make([]byte, length)

	if _, err = rand.Read(key); err != nil {
		return nil, err
	}

	return key, nil
}
