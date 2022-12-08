/*
Copyright 2017 The Kubernetes Authors.

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

// Package aes transforms values for storage at rest using AES-GCM.
package encryption

import (
	"context"
	"crypto/cipher"
	"crypto/rand"
	"fmt"
)

/*
Copied from:
	- "k8s.io/apiserver/pkg/storage/value"
	- "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
*/

// Transformer allows a value to be transformed before being read from or written to the underlying store. The methods
// must be able to undo the transformation caused by the other.
type Transformer interface {
	// TransformFromStorage may transform the provided data from its underlying storage representation or return an error.
	// Stale is true if the object on disk is stale and a write to etcd should be issued, even if the contents of the object
	// have not changed.
	TransformFromStorage(ctx context.Context, data []byte, dataCtx Context) (out []byte, stale bool, err error)
	// TransformToStorage may transform the provided data into the appropriate form in storage or return an error.
	TransformToStorage(ctx context.Context, data []byte, dataCtx Context) (out []byte, err error)
}

// Context is additional information that a storage transformation may need to verify the data at rest.
type Context interface {
	// AuthenticatedData should return an array of bytes that describes the current value. If the value changes,
	// the transformer may report the value as unreadable or tampered. This may be nil if no such description exists
	// or is needed. For additional verification, set this to data that strongly identifies the value, such as
	// the key and creation version of the stored data.
	AuthenticatedData() []byte
}

// DefaultContext is a simple implementation of Context for a slice of bytes.
type DefaultContext []byte

// AuthenticatedData returns itself.
func (c DefaultContext) AuthenticatedData() []byte { return c }

// gcm implements AEAD encryption of the provided values given a cipher.Block algorithm.
// The authenticated data provided as part of the Context method must match when the same
// value is set to and loaded from storage. In order to ensure that values cannot be copied by
// an attacker from a location under their control, use characteristics of the storage location
// (such as the etcd key) as part of the authenticated data.
//
// Because this mode requires a generated IV and IV reuse is a known weakness of AES-GCM, keys
// must be rotated before a birthday attack becomes feasible. NIST SP 800-38D
// (http://csrc.nist.gov/publications/nistpubs/800-38D/SP-800-38D.pdf) recommends using the same
// key with random 96-bit nonces (the default nonce length) no more than 2^32 times, and
// therefore transformers using this implementation *must* ensure they allow for frequent key
// rotation. Future work should include investigation of AES-GCM-SIV as an alternative to
// random nonces.
type gcm struct {
	block cipher.Block
}

// NewGCMTransformer takes the given block cipher and performs encryption and decryption on the given
// data.
func NewGCMTransformer(block cipher.Block) Transformer {
	return &gcm{block: block}
}

func (t *gcm) TransformFromStorage(ctx context.Context, data []byte, dataCtx Context) ([]byte, bool, error) {
	aead, err := cipher.NewGCM(t.block)
	if err != nil {
		return nil, false, err
	}
	nonceSize := aead.NonceSize()
	if len(data) < nonceSize {
		return nil, false, fmt.Errorf("the stored data was shorter than the required size")
	}
	result, err := aead.Open(nil, data[:nonceSize], data[nonceSize:], dataCtx.AuthenticatedData())
	return result, false, err
}

func (t *gcm) TransformToStorage(ctx context.Context, data []byte, dataCtx Context) ([]byte, error) {
	aead, err := cipher.NewGCM(t.block)
	if err != nil {
		return nil, err
	}
	nonceSize := aead.NonceSize()
	result := make([]byte, nonceSize+aead.Overhead()+len(data))
	n, err := rand.Read(result[:nonceSize])
	if err != nil {
		return nil, err
	}
	if n != nonceSize {
		return nil, fmt.Errorf("unable to read sufficient random bytes")
	}
	cipherText := aead.Seal(result[nonceSize:nonceSize], result[:nonceSize], data, dataCtx.AuthenticatedData())
	return result[:nonceSize+len(cipherText)], nil
}
