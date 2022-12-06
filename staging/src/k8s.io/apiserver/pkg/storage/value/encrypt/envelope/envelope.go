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
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"time"

	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope/metrics"
	"k8s.io/utils/lru"

	"golang.org/x/crypto/cryptobyte"
)

func init() {
	value.RegisterMetrics()
	metrics.RegisterMetrics()
}

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

	cacheSize    int
	cacheEnabled bool
}

// NewEnvelopeTransformer returns a transformer which implements a KEK-DEK based envelope encryption scheme.
// It uses envelopeService to encrypt and decrypt DEKs. Respective DEKs (in encrypted form) are prepended to
// the data items they encrypt. A cache (of size cacheSize) is maintained to store the most recently
// used decrypted DEKs in memory.
func NewEnvelopeTransformer(envelopeService Service, cacheSize int, baseTransformerFunc func(cipher.Block) value.Transformer) value.Transformer {
	var (
		cache *lru.Cache
	)

	if cacheSize > 0 {
		cache = lru.New(cacheSize)
	}
	return &envelopeTransformer{
		envelopeService:     envelopeService,
		transformers:        cache,
		baseTransformerFunc: baseTransformerFunc,
		cacheEnabled:        cacheSize > 0,
		cacheSize:           cacheSize,
	}
}

// TransformFromStorage decrypts data encrypted by this transformer using envelope encryption.
func (t *envelopeTransformer) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, bool, error) {
	metrics.RecordArrival(metrics.FromStorageLabel, time.Now())

	// Read the 16 bit length-of-DEK encoded at the start of the encrypted DEK. 16 bits can
	// represent a maximum key length of 65536 bytes. We are using a 256 bit key, whose
	// length cannot fit in 8 bits (1 byte). Thus, we use 16 bits (2 bytes) to store the length.
	var encKey cryptobyte.String
	s := cryptobyte.String(data)
	if ok := s.ReadUint16LengthPrefixed(&encKey); !ok {
		return nil, false, fmt.Errorf("invalid data encountered by envelope transformer: failed to read uint16 length prefixed data")
	}

	encData := []byte(s)

	// Look up the decrypted DEK from cache or Envelope.
	transformer := t.getTransformer(encKey)
	if transformer == nil {
		if t.cacheEnabled {
			value.RecordCacheMiss()
		}
		key, err := t.envelopeService.Decrypt(encKey)
		if err != nil {
			// Do NOT wrap this err using fmt.Errorf() or similar functions
			// because this gRPC status error has useful error code when
			// record the metric.
			return nil, false, err
		}

		transformer, err = t.addTransformer(encKey, key)
		if err != nil {
			return nil, false, err
		}
	}

	return transformer.TransformFromStorage(ctx, encData, dataCtx)
}

// TransformToStorage encrypts data to be written to disk using envelope encryption.
func (t *envelopeTransformer) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, error) {
	metrics.RecordArrival(metrics.ToStorageLabel, time.Now())
	newKey, err := generateKey(32)
	if err != nil {
		return nil, err
	}

	encKey, err := t.envelopeService.Encrypt(newKey)
	if err != nil {
		// Do NOT wrap this err using fmt.Errorf() or similar functions
		// because this gRPC status error has useful error code when
		// record the metric.
		return nil, err
	}

	transformer, err := t.addTransformer(encKey, newKey)
	if err != nil {
		return nil, err
	}

	result, err := transformer.TransformToStorage(ctx, data, dataCtx)
	if err != nil {
		return nil, err
	}
	// Append the length of the encrypted DEK as the first 2 bytes.
	b := cryptobyte.NewBuilder(nil)
	b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
		b.AddBytes([]byte(encKey))
	})
	b.AddBytes(result)

	return b.Bytes()
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
	if t.cacheEnabled {
		t.transformers.Add(base64.StdEncoding.EncodeToString(encKey), transformer)
		metrics.RecordDekCacheFillPercent(float64(t.transformers.Len()) / float64(t.cacheSize))
	}
	return transformer, nil
}

// getTransformer fetches the transformer corresponding to encKey from cache, if it exists.
func (t *envelopeTransformer) getTransformer(encKey []byte) value.Transformer {
	if !t.cacheEnabled {
		return nil
	}

	_transformer, found := t.transformers.Get(base64.StdEncoding.EncodeToString(encKey))
	if found {
		return _transformer.(value.Transformer)
	}
	return nil
}

// generateKey generates a random key using system randomness.
func generateKey(length int) (key []byte, err error) {
	defer func(start time.Time) {
		value.RecordDataKeyGeneration(start, err)
	}(time.Now())
	key = make([]byte, length)
	if _, err = rand.Read(key); err != nil {
		return nil, err
	}

	return key, nil
}
