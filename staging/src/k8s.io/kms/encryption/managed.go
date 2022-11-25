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
	"crypto/rand"
	"errors"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
)

const (
	// cacheSize is set to 100 keys, which can be used to decrypt up to 200 million encrypted
	// files or data encrypted within nearly 2 years.
	cacheSize = 100
)

var (
	// ErrNoCipher means that there is no remote kms given and therefore the keys in use can't be protected.
	ErrNoCipher = errors.New("no remote encryption service was specified")
)

// ManagedCipher is a set of keys. Only one key is used for encryption at one
// time. New keys are created automatically, when hitting safety thresholds.
type ManagedCipher struct {
	lock sync.RWMutex

	remoteKMSID []byte
	remoteKMS   EncrypterDecrypter

	localKEK []byte
	keys     *cache
}

// EncrypterDecrypter is a default encryption / decryption interface with an ID
// to support remote state.
type EncrypterDecrypter interface {
	Encrypt(ctx context.Context, plaintext []byte) (keyID, ciphertext []byte, err error)
	Decrypt(ctx context.Context, keyID, ciphertext []byte) (plaintext []byte, err error)
}

// CurrentKeyID returns the currently assumed remote Key ID.
func (m *ManagedCipher) CurrentKeyID() []byte {
	return m.remoteKMSID
}

// NewManagedCipher returns a pointer to a ManagedCipher. It is initialized with
// a cache, a reference to an remote kms and does an initial encryption call to
// the remote cipher.
func NewManagedCipher(ctx context.Context, remoteKMS EncrypterDecrypter) (*ManagedCipher, error) {
	if remoteKMS == nil {
		klog.V(2).Infof("create managed cipher without remote encryption failed")
		return nil, ErrNoCipher
	}

	key, err := NewKey()
	if err != nil {
		return nil, err
	}

	cipher, err := NewAESGCM(key)
	if err != nil {
		klog.V(4).Infof("create new cipher: %w", err)
		return nil, err
	}

	keyID, encKey, err := remoteKMS.Encrypt(ctx, key)
	if err != nil {
		klog.V(2).Infof("encrypt with remote: %w", err)
		return nil, err
	}

	cache := newCache(cacheSize)
	cache.Add(encKey, cipher)

	mc := ManagedCipher{
		remoteKMS:   remoteKMS,
		remoteKMSID: keyID,

		keys:     cache,
		localKEK: encKey,
	}

	return &mc, nil
}

func (m *ManagedCipher) Run(ctx context.Context) {
	go wait.Until(func() {
		cipher, ok := m.keys.Get(m.localKEK)
		if ok {
			klog.Warning("KMS Plugin points to a local KEK that can't be found")
			return
		}

		if cipher.isReplaceable() {
			m.addNewKey(ctx)
		}
	}, time.Second*10, ctx.Done())
}

func (m *ManagedCipher) addNewKey(ctx context.Context) error {
	key, err := NewKey()
	if err != nil {
		return err
	}

	cipher, err := NewAESGCM(key)
	if err != nil {
		klog.V(4).Infof("create new cipher: %w", err)
		return err
	}

	keyID, encKey, err := m.remoteKMS.Encrypt(ctx, key)
	if err != nil {
		klog.Warning("encrypt with remote: %w", err)
		return err
	}

	m.keys.lruCache.Add(encKey, cipher)

	m.lock.Lock()
	{
		m.localKEK = encKey
		m.remoteKMSID = keyID
	}
	m.lock.Unlock()

	return nil
}

// Encrypt encrypts given plaintext and returns the key used in encrypted form.
// The encrypted key is encrypted by the given remote KMS.
func (m *ManagedCipher) Encrypt(ctx context.Context, pt []byte) ([]byte, []byte, []byte, error) {
	cipher, ok := m.keys.Get(m.localKEK)
	if !ok {
		klog.Warning("can't find local kek in cache")

		return nil, nil, nil, errors.New("local kek is not in cache")
	}

	// It can happen that cipher.Encrypt fails on an exhausted key. The probability is very low.
	ct, err := cipher.Encrypt(ctx, pt)
	if err != nil {
		klog.V(4).Infof("encrypt plaintext: %w", err)
		return nil, nil, nil, err
	}

	return m.remoteKMSID, m.localKEK, ct, nil
}

// DecryptRemotely decrypts given ciphertext by sending it directly to the
// remote kms.
func (m *ManagedCipher) DecryptRemotely(ctx context.Context, id, ct []byte) ([]byte, error) {
	return m.remoteKMS.Decrypt(ctx, id, ct)
}

// Decrypt decrypts the given ciphertext. If the given encrypted key is unknown,
// Remote KMS is asked for decryption of the encrypted key.
func (m *ManagedCipher) Decrypt(ctx context.Context, keyID, encKey, ct []byte) ([]byte, error) {
	// Lookup key from cache.
	cipher, ok := m.keys.Get(encKey)
	if ok {
		pt, err := cipher.Decrypt(ctx, ct)
		if err != nil {
			klog.V(4).Infof("decrypt ciphertext: %w", err)
			return nil, err
		}

		return pt, nil
	}

	// not in cache flow
	klog.V(4).Infof("key has no value in cache")

	// plainKey is a plaintext key and should be handled cautiously.
	plainKey, err := m.remoteKMS.Decrypt(ctx, keyID, encKey)
	if err != nil {
		klog.Warning("decrypt key by remote:", err)

		return nil, err
	}

	// Set up the cipher for the given key.
	cipher, err = FromKey(plainKey)
	if err != nil {
		klog.V(4).Infof("use key for encryption: %w", err)

		return nil, err
	}

	// Add to cache.
	m.keys.Add(encKey, cipher)
	klog.V(4).Infof("key from ciphertext added to cache")

	// Eventually decrypt with new key.
	pt, err := cipher.Decrypt(ctx, ct)
	if err != nil {
		klog.V(4).Infof("decrypt ciphertext: %w", err)
		return nil, err
	}

	return pt, nil
}

func NewKey() ([]byte, error) {
	return randomBytes(keySize)
}

// randomBytes generates length amount of bytes.
func randomBytes(length int) (key []byte, err error) {
	key = make([]byte, length)

	if _, err = rand.Read(key); err != nil {
		return nil, err
	}

	return key, nil
}
