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

// Package halfconn manages the inbound or outbound traffic of a TLS 1.3
// connection.
package halfconn

import (
	"fmt"
	"sync"

	s2apb "github.com/google/s2a-go/internal/proto/common_go_proto"
	"github.com/google/s2a-go/internal/record/internal/aeadcrypter"
	"golang.org/x/crypto/cryptobyte"
)

// The constants below were taken from Section 7.2 and 7.3 in
// https://tools.ietf.org/html/rfc8446#section-7. They are used as the label
// in HKDF-Expand-Label.
const (
	tls13Key    = "tls13 key"
	tls13Nonce  = "tls13 iv"
	tls13Update = "tls13 traffic upd"
)

// S2AHalfConnection stores the state of the TLS 1.3 connection in the
// inbound or outbound direction.
type S2AHalfConnection struct {
	cs       ciphersuite
	expander hkdfExpander
	// mutex guards sequence, aeadCrypter, trafficSecret, and nonce.
	mutex         sync.Mutex
	aeadCrypter   aeadcrypter.S2AAEADCrypter
	sequence      counter
	trafficSecret []byte
	nonce         []byte
}

// New creates a new instance of S2AHalfConnection given a ciphersuite and a
// traffic secret.
func New(ciphersuite s2apb.Ciphersuite, trafficSecret []byte, sequence uint64) (*S2AHalfConnection, error) {
	cs, err := newCiphersuite(ciphersuite)
	if err != nil {
		return nil, fmt.Errorf("failed to create new ciphersuite: %v", ciphersuite)
	}
	if cs.trafficSecretSize() != len(trafficSecret) {
		return nil, fmt.Errorf("supplied traffic secret must be %v bytes, given: %v bytes", cs.trafficSecretSize(), len(trafficSecret))
	}

	hc := &S2AHalfConnection{cs: cs, expander: newDefaultHKDFExpander(cs.hashFunction()), sequence: newCounter(sequence), trafficSecret: trafficSecret}
	if err = hc.updateCrypterAndNonce(hc.trafficSecret); err != nil {
		return nil, fmt.Errorf("failed to create half connection using traffic secret: %v", err)
	}

	return hc, nil
}

// Encrypt encrypts the plaintext and computes the tag of dst and plaintext.
// dst and plaintext may fully overlap or not at all. Note that the sequence
// number will still be incremented on failure, unless the sequence has
// overflowed.
func (hc *S2AHalfConnection) Encrypt(dst, plaintext, aad []byte) ([]byte, error) {
	hc.mutex.Lock()
	sequence, err := hc.getAndIncrementSequence()
	if err != nil {
		hc.mutex.Unlock()
		return nil, err
	}
	nonce := hc.maskedNonce(sequence)
	crypter := hc.aeadCrypter
	hc.mutex.Unlock()
	return crypter.Encrypt(dst, plaintext, nonce, aad)
}

// Decrypt decrypts ciphertext and verifies the tag. dst and ciphertext may
// fully overlap or not at all. Note that the sequence number will still be
// incremented on failure, unless the sequence has overflowed.
func (hc *S2AHalfConnection) Decrypt(dst, ciphertext, aad []byte) ([]byte, error) {
	hc.mutex.Lock()
	sequence, err := hc.getAndIncrementSequence()
	if err != nil {
		hc.mutex.Unlock()
		return nil, err
	}
	nonce := hc.maskedNonce(sequence)
	crypter := hc.aeadCrypter
	hc.mutex.Unlock()
	return crypter.Decrypt(dst, ciphertext, nonce, aad)
}

// UpdateKey advances the traffic secret key, as specified in
// https://tools.ietf.org/html/rfc8446#section-7.2. In addition, it derives
// a new key and nonce, and resets the sequence number.
func (hc *S2AHalfConnection) UpdateKey() error {
	hc.mutex.Lock()
	defer hc.mutex.Unlock()

	var err error
	hc.trafficSecret, err = hc.deriveSecret(hc.trafficSecret, []byte(tls13Update), hc.cs.trafficSecretSize())
	if err != nil {
		return fmt.Errorf("failed to derive traffic secret: %v", err)
	}

	if err = hc.updateCrypterAndNonce(hc.trafficSecret); err != nil {
		return fmt.Errorf("failed to update half connection: %v", err)
	}

	hc.sequence.reset()
	return nil
}

// TagSize returns the tag size in bytes of the underlying AEAD crypter.
func (hc *S2AHalfConnection) TagSize() int {
	return hc.aeadCrypter.TagSize()
}

// updateCrypterAndNonce takes a new traffic secret and updates the crypter
// and nonce. Note that the mutex must be held while calling this function.
func (hc *S2AHalfConnection) updateCrypterAndNonce(newTrafficSecret []byte) error {
	key, err := hc.deriveSecret(newTrafficSecret, []byte(tls13Key), hc.cs.keySize())
	if err != nil {
		return fmt.Errorf("failed to update key: %v", err)
	}

	hc.nonce, err = hc.deriveSecret(newTrafficSecret, []byte(tls13Nonce), hc.cs.nonceSize())
	if err != nil {
		return fmt.Errorf("failed to update nonce: %v", err)
	}

	hc.aeadCrypter, err = hc.cs.aeadCrypter(key)
	if err != nil {
		return fmt.Errorf("failed to update AEAD crypter: %v", err)
	}
	return nil
}

// getAndIncrement returns the current sequence number and increments it. Note
// that the mutex must be held while calling this function.
func (hc *S2AHalfConnection) getAndIncrementSequence() (uint64, error) {
	sequence, err := hc.sequence.value()
	if err != nil {
		return 0, err
	}
	hc.sequence.increment()
	return sequence, nil
}

// maskedNonce creates a copy of the nonce that is masked with the sequence
// number. Note that the mutex must be held while calling this function.
func (hc *S2AHalfConnection) maskedNonce(sequence uint64) []byte {
	const uint64Size = 8
	nonce := make([]byte, len(hc.nonce))
	copy(nonce, hc.nonce)
	for i := 0; i < uint64Size; i++ {
		nonce[aeadcrypter.NonceSize-uint64Size+i] ^= byte(sequence >> uint64(56-uint64Size*i))
	}
	return nonce
}

// deriveSecret implements the Derive-Secret function, as specified in
// https://tools.ietf.org/html/rfc8446#section-7.1.
func (hc *S2AHalfConnection) deriveSecret(secret, label []byte, length int) ([]byte, error) {
	var hkdfLabel cryptobyte.Builder
	hkdfLabel.AddUint16(uint16(length))
	hkdfLabel.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
		b.AddBytes(label)
	})
	// Append an empty `Context` field to the label, as specified in the RFC.
	// The half connection does not use the `Context` field.
	hkdfLabel.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
		b.AddBytes([]byte(""))
	})
	hkdfLabelBytes, err := hkdfLabel.Bytes()
	if err != nil {
		return nil, fmt.Errorf("deriveSecret failed: %v", err)
	}
	return hc.expander.expand(secret, hkdfLabelBytes, length)
}
