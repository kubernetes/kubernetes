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

// Package envelope transforms values for storage at rest using a Envelope provider
package envelope

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"encoding/binary"
	"fmt"

	"k8s.io/apiserver/pkg/storage/value"

	lru "github.com/hashicorp/golang-lru"
)

// defaultCacheSize is the number of decrypted DEKs which would be cached by the transformer.
const defaultCacheSize = 1000

// Service allows encrypting and decrypting data using an external Key Management Service.
type Service interface {
	// Decrypt a given bytearray to obtain the original data as bytes.
	Decrypt(data []byte) ([]byte, error)
	// Encrypt bytes to a ciphertext.
	Encrypt(data []byte) ([]byte, error)
}

type envelopeTransformer struct {
	envelopeService Service

	// transformers is a thread-safe LRU cache which caches decrypted DEKs indexed by their encrypted form.
	transformers *lru.Cache

	// baseTransformerFunc creates a new transformer for encrypting the data with the DEK.
	baseTransformerFunc func(cipher.Block) value.Transformer
}

// NewEnvelopeTransformer returns a transformer which implements a KEK-DEK based envelope encryption scheme.
// It uses envelopeService to encrypt and decrypt DEKs. Respective DEKs (in encrypted form) are prepended to
// the data items they encrypt. A cache (of size cacheSize) is maintained to store the most recently
// used decrypted DEKs in memory.
func NewEnvelopeTransformer(envelopeService Service, cacheSize int, baseTransformerFunc func(cipher.Block) value.Transformer) (value.Transformer, error) {
	if cacheSize == 0 {
		cacheSize = defaultCacheSize
	}
	cache, err := lru.New(cacheSize)
	if err != nil {
		return nil, err
	}
	return &envelopeTransformer{
		envelopeService:     envelopeService,
		transformers:        cache,
		baseTransformerFunc: baseTransformerFunc,
	}, nil
}

// TransformFromStorage decrypts data encrypted by this transformer using envelope encryption.
func (t *envelopeTransformer) TransformFromStorage(data []byte, context value.Context) ([]byte, bool, error) {
	// Read the 16 bit length-of-DEK encoded at the start of the encrypted DEK. 16 bits can
	// represent a maximum key length of 65536 bytes. We are using a 256 bit key, whose
	// length cannot fit in 8 bits (1 byte). Thus, we use 16 bits (2 bytes) to store the length.
	keyLen := int(binary.BigEndian.Uint16(data[:2]))
	if keyLen+2 > len(data) {
		return nil, false, fmt.Errorf("invalid data encountered by genvelope transformer, length longer than available bytes: %q", data)
	}
	encKey := data[2 : keyLen+2]
	encData := data[2+keyLen:]

	// Look up the decrypted DEK from cache or Envelope.
	transformer := t.getTransformer(encKey)
	if transformer == nil {
		key, err := t.envelopeService.Decrypt(encKey)
		if err != nil {
			return nil, false, fmt.Errorf("error while decrypting key: %q", err)
		}
		transformer, err = t.addTransformer(encKey, key)
		if err != nil {
			return nil, false, err
		}
	}
	return transformer.TransformFromStorage(encData, context)
}

// TransformToStorage encrypts data to be written to disk using envelope encryption.
func (t *envelopeTransformer) TransformToStorage(data []byte, context value.Context) ([]byte, error) {
	newKey, err := generateKey(32)
	if err != nil {
		return nil, err
	}

	encKey, err := t.envelopeService.Encrypt(newKey)
	if err != nil {
		return nil, err
	}

	transformer, err := t.addTransformer(encKey, newKey)
	if err != nil {
		return nil, err
	}

	// Append the length of the encrypted DEK as the first 2 bytes.
	encKeyLen := make([]byte, 2)
	encKeyBytes := []byte(encKey)
	binary.BigEndian.PutUint16(encKeyLen, uint16(len(encKeyBytes)))

	prefix := append(encKeyLen, encKeyBytes...)

	prefixedData := make([]byte, len(prefix), len(data)+len(prefix))
	copy(prefixedData, prefix)
	result, err := transformer.TransformToStorage(data, context)
	if err != nil {
		return nil, err
	}
	prefixedData = append(prefixedData, result...)
	return prefixedData, nil
}

var _ value.Transformer = &envelopeTransformer{}

// addTransformer inserts a new transformer to the Envelope cache of DEKs for future reads.
func (t *envelopeTransformer) addTransformer(encKey []byte, key []byte) (value.Transformer, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	transformer := t.baseTransformerFunc(block)
	// Use base64 of encKey as the key into the cache because hashicorp/golang-lru
	// cannot hash []uint8.
	t.transformers.Add(base64.StdEncoding.EncodeToString(encKey), transformer)
	return transformer, nil
}

// getTransformer fetches the transformer corresponding to encKey from cache, if it exists.
func (t *envelopeTransformer) getTransformer(encKey []byte) value.Transformer {
	_transformer, found := t.transformers.Get(base64.StdEncoding.EncodeToString(encKey))
	if found {
		return _transformer.(value.Transformer)
	}
	return nil
}

// generateKey generates a random key using system randomness.
func generateKey(length int) ([]byte, error) {
	key := make([]byte, length)
	_, err := rand.Read(key)
	if err != nil {
		return nil, err
	}

	return key, nil
}
