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

package aes

import (
	"bytes"
	"hash/maphash"
	"time"
	"unsafe"

	utilcache "k8s.io/apimachinery/pkg/util/cache"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/utils/clock"
)

// cacheShards is the number of independent Expiring caches the KDF cache is split into.
// Every decrypt reads its shard under a shared lock, but a miss is followed by a Set, which takes
// the lock exclusively and runs GC inline. With a single shard, the Sets for each new object
// version stall every other decrypt under the same seed, which under high watch fan-out is
// effectively every decrypt in the apiserver.
const cacheShards = 128

type simpleCache struct {
	shards []*utilcache.Expiring
	seed   maphash.Seed
	ttl    time.Duration
}

func newSimpleCache(clock clock.Clock, ttl time.Duration) *simpleCache {
	return newSimpleCacheWithShards(clock, ttl, cacheShards)
}

func newSimpleCacheWithShards(clock clock.Clock, ttl time.Duration, shards int) *simpleCache {
	c := &simpleCache{shards: make([]*utilcache.Expiring, shards), seed: maphash.MakeSeed(), ttl: ttl}
	for i := range c.shards {
		cache := utilcache.NewExpiringWithClock(clock)
		// "Stale" entries are always valid for us because the TTL is just used to prevent
		// unbounded growth on the cache - for a given info the transformer is always the same.
		// The key always corresponds to the exact same value, with the caveat that
		// since we use the value.Context.AuthenticatedData to overwrite old keys,
		// we always have to check that the info matches (to validate the transformer is correct).
		cache.AllowExpiredGet = true
		c.shards[i] = cache
	}
	return c
}

func (c *simpleCache) shard(key string) *utilcache.Expiring {
	return c.shards[maphash.String(c.seed, key)%uint64(len(c.shards))]
}

// Len returns the number of entries across all shards.
func (c *simpleCache) Len() int {
	n := 0
	for _, s := range c.shards {
		n += s.Len()
	}
	return n
}

// given a key, return the transformer, or nil if it does not exist in the cache
func (c *simpleCache) get(info []byte, dataCtx value.Context) *transformerWithInfo {
	key := keyFunc(dataCtx)
	val, ok := c.shard(key).Get(key)
	if !ok {
		return nil
	}

	transformer := val.(*transformerWithInfo)

	if !bytes.Equal(transformer.info, info) {
		return nil
	}

	return transformer
}

// set caches the record for the key
func (c *simpleCache) set(dataCtx value.Context, transformer *transformerWithInfo) {
	if dataCtx == nil || len(dataCtx.AuthenticatedData()) == 0 {
		panic("authenticated data must not be empty")
	}
	if transformer == nil {
		panic("transformer must not be nil")
	}
	if len(transformer.info) == 0 {
		panic("info must not be empty")
	}
	key := keyFunc(dataCtx)
	shard := c.shard(key)
	// Many watchers decrypt the same new object version at nearly the same time and all miss.
	// Only the first needs to store it; the rest find it under the shared read lock and skip
	// the exclusive Set (and its inline GC).
	if val, ok := shard.Get(key); ok && bytes.Equal(val.(*transformerWithInfo).info, transformer.info) {
		return
	}
	shard.Set(key, transformer, c.ttl)
}

func keyFunc(dataCtx value.Context) string {
	return toString(dataCtx.AuthenticatedData())
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
