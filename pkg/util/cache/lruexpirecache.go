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

	"github.com/golang/groupcache/lru"
)

// Clock defines an interface for obtaining the current time
type Clock interface {
	Now() time.Time
}

// realClock implements the Clock interface by calling time.Now()
type realClock struct{}

func (realClock) Now() time.Time { return time.Now() }

type LRUExpireCache struct {
	// clock is used to obtain the current time
	clock Clock

	cache *lru.Cache
	lock  sync.Mutex
}

// NewLRUExpireCache creates an expiring cache with the given size
func NewLRUExpireCache(maxSize int) *LRUExpireCache {
	return &LRUExpireCache{clock: realClock{}, cache: lru.New(maxSize)}
}

// NewLRUExpireCache creates an expiring cache with the given size, using the specified clock to obtain the current time
func NewLRUExpireCacheWithClock(maxSize int, clock Clock) *LRUExpireCache {
	return &LRUExpireCache{clock: clock, cache: lru.New(maxSize)}
}

type cacheEntry struct {
	value      interface{}
	expireTime time.Time
}

func (c *LRUExpireCache) Add(key lru.Key, value interface{}, ttl time.Duration) {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.cache.Add(key, &cacheEntry{value, c.clock.Now().Add(ttl)})
	// Remove entry from cache after ttl.
	time.AfterFunc(ttl, func() { c.remove(key) })
}

func (c *LRUExpireCache) Get(key lru.Key) (interface{}, bool) {
	c.lock.Lock()
	defer c.lock.Unlock()
	e, ok := c.cache.Get(key)
	if !ok {
		return nil, false
	}
	if c.clock.Now().After(e.(*cacheEntry).expireTime) {
		go c.remove(key)
		return nil, false
	}
	return e.(*cacheEntry).value, true
}

func (c *LRUExpireCache) remove(key lru.Key) {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.cache.Remove(key)
}
