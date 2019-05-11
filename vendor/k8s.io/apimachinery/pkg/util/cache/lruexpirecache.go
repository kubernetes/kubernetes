/*
Copyright 2016 The Kubernetes Authors.

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

package cache

import (
	"sync"
	"time"

	"github.com/hashicorp/golang-lru"
)

// Clock defines an interface for obtaining the current time
type Clock interface {
	Now() time.Time
}

// realClock implements the Clock interface by calling time.Now()
type realClock struct{}

func (realClock) Now() time.Time { return time.Now() }

// LRUExpireCache is a cache that ensures the mostly recently accessed keys are returned with
// a ttl beyond which keys are forcibly expired.
type LRUExpireCache struct {
	// clock is used to obtain the current time
	clock Clock

	cache *lru.Cache
	lock  sync.Mutex
}

// NewLRUExpireCache creates an expiring cache with the given size
func NewLRUExpireCache(maxSize int) *LRUExpireCache {
	return NewLRUExpireCacheWithClock(maxSize, realClock{})
}

// NewLRUExpireCacheWithClock creates an expiring cache with the given size, using the specified clock to obtain the current time.
func NewLRUExpireCacheWithClock(maxSize int, clock Clock) *LRUExpireCache {
	cache, err := lru.New(maxSize)
	if err != nil {
		// if called with an invalid size
		panic(err)
	}
	return &LRUExpireCache{clock: clock, cache: cache}
}

type cacheEntry struct {
	value      interface{}
	expireTime time.Time
}

// Add adds the value to the cache at key with the specified maximum duration.
func (c *LRUExpireCache) Add(key interface{}, value interface{}, ttl time.Duration) {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.cache.Add(key, &cacheEntry{value, c.clock.Now().Add(ttl)})
}

// Get returns the value at the specified key from the cache if it exists and is not
// expired, or returns false.
func (c *LRUExpireCache) Get(key interface{}) (interface{}, bool) {
	c.lock.Lock()
	defer c.lock.Unlock()
	e, ok := c.cache.Get(key)
	if !ok {
		return nil, false
	}
	if c.clock.Now().After(e.(*cacheEntry).expireTime) {
		c.cache.Remove(key)
		return nil, false
	}
	return e.(*cacheEntry).value, true
}

// Remove removes the specified key from the cache if it exists
func (c *LRUExpireCache) Remove(key interface{}) {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.cache.Remove(key)
}

// Keys returns all the keys in the cache, even if they are expired. Subsequent calls to
// get may return not found. It returns all keys from oldest to newest.
func (c *LRUExpireCache) Keys() []interface{} {
	c.lock.Lock()
	defer c.lock.Unlock()
	return c.cache.Keys()
}
