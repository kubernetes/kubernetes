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
	"k8s.io/utils/clock"
)

type simpleCache struct {
	cache *utilcache.Expiring
	ttl   time.Duration
	// hashPool is a per cache pool of hash.Hash (to avoid allocations from building the Hash)
	// SHA-256 is used to prevent collisions
	hashPool *sync.Pool
}

func newSimpleCache(clock clock.Clock, ttl time.Duration) *simpleCache {
	return &simpleCache{
		cache: utilcache.NewExpiringWithClock(clock),
		ttl:   ttl,
		hashPool: &sync.Pool{
			New: func() interface{} {
				return sha256.New()
			},
		},
	}
}

// given a key, return the transformer, or nil if it does not exist in the cache
func (c *simpleCache) get(key []byte) value.Transformer {
	record, ok := c.cache.Get(c.keyFunc(key))
	if !ok {
		return nil
	}
	return record.(value.Transformer)
}

// set caches the record for the key
func (c *simpleCache) set(key []byte, transformer value.Transformer) {
	if len(key) == 0 {
		panic("key must not be empty")
	}
	if transformer == nil {
		panic("transformer must not be nil")
	}
	c.cache.Set(c.keyFunc(key), transformer, c.ttl)
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
	return *(*string)(unsafe.Pointer(&b))
}
