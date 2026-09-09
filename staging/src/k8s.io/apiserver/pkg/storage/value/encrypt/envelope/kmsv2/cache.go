/*
Copyright 2023 The Kubernetes Authors.

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

// Package kmsv2 transforms values for storage at rest using a Envelope v2 provider
package kmsv2

import (
	"crypto/sha256"
	"hash"
	"sync"
	"time"
	"unsafe"

	utilcache "k8s.io/apimachinery/pkg/util/cache"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope/metrics"
	"k8s.io/utils/clock"
)

// simpleCache stores the decryption subset of value.Transformer (value.Read).
// this statically enforces that transformers placed in the cache are not used for encryption.
// this is relevant in the context of nonce collision since transformers that are created
// from encrypted DEKs retrieved from etcd cannot maintain their nonce counter state.
type simpleCache struct {
	cache *utilcache.Expiring
	ttl   time.Duration
	// hashPool is a per cache pool of hash.Hash (to avoid allocations from building the Hash)
	// SHA-256 is used to prevent collisions
	hashPool        *sync.Pool
	providerName    string
	mu              sync.Mutex                          // guards call to set
	recordCacheSize func(providerName string, size int) // for unit tests
}

func newSimpleCache(clock clock.Clock, ttl time.Duration, providerName string) *simpleCache {
	cache := utilcache.NewExpiringWithClock(clock)
	cache.AllowExpiredGet = true // for a given key, the value (the decryptTransformer) is always the same
	return &simpleCache{
		cache: cache,
		ttl:   ttl,
		hashPool: &sync.Pool{
			New: func() interface{} {
				return sha256.New()
			},
		},
		providerName:    providerName,
		recordCacheSize: metrics.RecordDekSourceCacheSize,
	}
}

// given a key, return the transformer, or nil if it does not exist in the cache
func (c *simpleCache) get(key []byte) value.Read {
	record, ok := c.cache.Get(c.keyFunc(key))
	if !ok {
		return nil
	}
	return record.(value.Read)
}

// set caches the record for the key
func (c *simpleCache) set(key []byte, transformer value.Read) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(key) == 0 {
		panic("key must not be empty")
	}
	if transformer == nil {
		panic("transformer must not be nil")
	}
	c.cache.Set(c.keyFunc(key), transformer, c.ttl)
	// Add metrics for cache size
	c.recordCacheSize(c.providerName, c.cache.Len())
}

// keyFunc generates a string key by hashing the inputs.
// This lowers the memory requirement of the cache.
func (c *simpleCache) keyFunc(s []byte) string {
	h := c.hashPool.Get().(hash.Hash)
	h.Reset()

	if _, err := h.Write(s); err != nil {
		panic(err) // Write() on hash never fails
	}
	key := toString(h.Sum(nil)) // skip base64 encoding to save an allocation
	c.hashPool.Put(h)

	return key
}

// toString performs unholy acts to avoid allocations
func toString(b []byte) string {
	// unsafe.SliceData relies on cap whereas we want to rely on len
	if len(b) == 0 {
		return ""
	}
	// Copied from go 1.20.1 strings.Builder.String
	// https://github.com/golang/go/blob/202a1a57064127c3f19d96df57b9f9586145e21c/src/strings/builder.go#L48
	return unsafe.String(unsafe.SliceData(b), len(b))
}
