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
	"encoding/base64"
	"errors"
	"fmt"
	"sync"
	"time"

	"k8s.io/klog/v2"
)

const (
	// MaxUsage with 2^21 is a very defensive value. 2^32 is more commonly used.
	MaxUsage = 2097151
	// nonceSize is the size of the nonce. Do not change, without breaking version change.
	// The nonceSize is a de facto standard.
	nonceSize = 12
	// cacheSize is set to 100 keys, which can be used to decrypt up to 200 million encrypted
	// files or data encrypted within nearly 2 years.
	cacheSize = 100
)

var (
	// ErrKeyExpired means that the expiration time of a key has come and it shouldn't be used any more.
	ErrKeyExpired = errors.New("key is out of date and shouldn't be used anymore for encryption")
	// ErrNoCipher means that there is no remote kms given and therefore the keys in use can't be protected.
	ErrNoCipher = errors.New("no remote encryption service was specified")

	week = time.Hour * 24 * 7
)

// ManagedCipher is a set of keys. Only one key is used for encryption at one
// time. New keys are created automatically, when hitting safety thresholds.
type ManagedCipher struct {
	counter uint32
	expires time.Time

	keys *cache

	currentRemoteKMSID []byte
	currentLocalKEK    []byte

	fallbackRemoteKMSID []byte
	fallbackLocalKEK    []byte

	remoteKMS EncrypterDecrypter

	m sync.Mutex
}

// EncrypterDecrypter is a default encryption / decryption interface with an ID
// to support remote state.
type EncrypterDecrypter interface {
	// TODO: consider adding context to interface.
	Encrypt(plaintext []byte) (keyID, ciphertext []byte, err error)
	Decrypt(keyID, ciphertext []byte) (plaintext []byte, err error)
}

// CurrentKeyID returns the currently assumed remote Key ID.
func (m *ManagedCipher) CurrentKeyID() []byte {
	return m.currentRemoteKMSID
}

// NewManagedCipher returns a pointer to a ManagedCipher. It is initialized with
// a cache, a reference to an remote cipher (like a KMS service or HMS) and
// does an initial encryption call to the remote cipher.
func NewManagedCipher(remoteCipher EncrypterDecrypter) (*ManagedCipher, error) {
	if remoteCipher == nil {
		klog.Infof("create managed cipher without remote encryption failed")
		return nil, ErrNoCipher
	}

	cipher, err := NewAESGCM()
	if err != nil {
		klog.Infof("create new cipher: %w", err)
		return nil, err
	}

	keyID, encCipher, err := remoteCipher.Encrypt(cipher.Key())
	if err != nil {
		klog.Infof("encrypt with remote: %w", err)
		return nil, err
	}

	cache := newCache(cacheSize)
	cache.Add(encCipher, cipher)

	mc := ManagedCipher{
		keys:               cache,
		counter:            0,
		expires:            time.Now().Add(week),
		remoteKMS:          remoteCipher,
		currentRemoteKMSID: keyID,
		currentLocalKEK:    encCipher,
	}

	klog.Infof("new managed cipher is created")

	go func() {
		if err := mc.addFallbackCipher(); err != nil {
			klog.Infof("set up fallback cipher: %w", err)
		}
	}()

	return &mc, nil
}

func (m *ManagedCipher) addFallbackCipher() error {
	cipher, err := NewAESGCM()
	if err != nil {
		klog.Infof("create new currentCipher: %w", err)
		return err
	}

	keyID, encCipher, err := m.remoteKMS.Encrypt(cipher.Key())
	if err != nil {
		klog.Infof("encrypt with remote: %w", err)
		return err
	}

	m.keys.Add(encCipher, cipher)
	m.fallbackLocalKEK = encCipher
	m.fallbackRemoteKMSID = keyID

	return nil
}

func (m *ManagedCipher) manageKey() error {
	m.m.Lock()
	defer m.m.Unlock() // should be bound to main thread

	// If the key is safe to use, do nothing.
	m.counter = m.counter + 1
	if m.counter < MaxUsage && time.Now().Before(m.expires) {
		return nil
	}

	// If fallback cipher doesn't exist, create one synchronously.
	if m.fallbackLocalKEK == nil {
		// In case that an error happened, while setting the fallback cipher
		// asynchronously, do it now synchronously.
		if err := m.addFallbackCipher(); err != nil {
			return err
		}
	}

	// Switch from current to fallback cipher
	m.currentRemoteKMSID = m.fallbackRemoteKMSID
	m.fallbackRemoteKMSID = nil
	m.currentLocalKEK = m.fallbackLocalKEK
	m.fallbackLocalKEK = nil

	// Reset checks.
	m.expires = time.Now().Add(week)
	m.counter = 0

	go func() {
		// Add fallback cipher asynchronously and optimistically.
		if err := m.addFallbackCipher(); err != nil {
			klog.Infof("set up fallback cipher: %w", err)
		}
	}()

	return nil
}

// Encrypt encrypts given plaintext and returns the key used in encrypted form.
// The encrypted key is encrypted by the given remote KMS.
func (m *ManagedCipher) Encrypt(ctx context.Context, pt []byte) ([]byte, []byte, []byte, error) {
	if err := m.manageKey(); err != nil {
		return nil, nil, nil, fmt.Errorf("manage keys upfront of an encryption: %w", err)
	}

	cipher, ok := m.keys.Get(m.currentLocalKEK)
	if !ok {
		klog.Infof(
			"current plugin key (%q) has no value in cache",
			base64.StdEncoding.EncodeToString(m.currentLocalKEK),
		)
		return nil, nil, nil, fmt.Errorf(
			"plugin is broken, current key is unknown (%q)",
			base64.StdEncoding.EncodeToString(m.currentLocalKEK),
		)
	}

	ct, err := cipher.Encrypt(ctx, pt)
	if err != nil {
		klog.Infof("encrypt plaintext: %w", err)
		return nil, nil, nil, err
	}

	return m.currentRemoteKMSID, m.currentLocalKEK, ct, nil
}

// DecryptRemotely decrypts given ciphertext by sending it directly to the
// remote kms.
func (m *ManagedCipher) DecryptRemotely(id, ct []byte) ([]byte, error) {
	return m.remoteKMS.Decrypt(id, ct)
}

// Decrypt decrypts the given ciphertext. If the given encrypted key is unknown,
// Remote KMS is asked for decryption of the encrypted key.
func (m *ManagedCipher) Decrypt(ctx context.Context, keyID, encKey, ct []byte) ([]byte, error) {
	// Lookup key from cache.
	cipher, ok := m.keys.Get(encKey)
	if ok {
		pt, err := cipher.Decrypt(ctx, ct)
		if err != nil {
			klog.Infof("decrypt ciphertext: %w", err)
			return nil, err
		}

		return pt, nil
	}

	// not in cache flow

	klog.Infof(
		"key (%q) has no value in cache",
		base64.StdEncoding.EncodeToString(encKey),
	)

	// plainKey is a plaintext key and should be handled cautiously.
	plainKey, err := m.remoteKMS.Decrypt(keyID, encKey)
	if err != nil {
		klog.Infof(
			"decrypt key (%q) by remote:",
			base64.StdEncoding.EncodeToString(encKey),
			err,
		)

		return nil, err
	}

	// Set up the cipher for the given key.
	cipher, err = FromKey(plainKey)
	if err != nil {
		klog.Infof(
			"use key (%q) for encryption: %w",
			base64.StdEncoding.EncodeToString(encKey),
			err,
		)
		return nil, err
	}

	// Add to cache.
	m.keys.Add(encKey, cipher)
	klog.Infof(
		"key (%q) from ciphertext added to cache",
		base64.StdEncoding.EncodeToString(encKey),
	)

	// Eventually decrypt with new key.
	pt, err := cipher.Decrypt(ctx, ct)
	if err != nil {
		klog.Infof("decrypt ciphertext: %w", err)
		return nil, err
	}

	return pt, nil
}
