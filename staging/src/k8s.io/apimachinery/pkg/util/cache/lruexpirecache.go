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
	"container/list"
	"sync"
	"time"
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

	lock sync.Mutex

	maxSize      int
	evictionList list.List
	entries      map[interface{}]*list.Element
}

// NewLRUExpireCache creates an expiring cache with the given size
func NewLRUExpireCache(maxSize int) *LRUExpireCache {
	return NewLRUExpireCacheWithClock(maxSize, realClock{})
}

// NewLRUExpireCacheWithClock creates an expiring cache with the given size, using the specified clock to obtain the current time.
func NewLRUExpireCacheWithClock(maxSize int, clock Clock) *LRUExpireCache {
	if maxSize <= 0 {
		panic("maxSize must be > 0")
	}

	return &LRUExpireCache{
		clock:   clock,
		maxSize: maxSize,
		entries: map[interface{}]*list.Element{},
	}
}

type cacheEntry struct {
	key        interface{}
	value      interface{}
	expireTime time.Time
}

// Add adds the value to the cache at key with the specified maximum duration.
func (c *LRUExpireCache) Add(key interface{}, value interface{}, ttl time.Duration) {
	c.lock.Lock()
	defer c.lock.Unlock()

	// Key already exists
	oldElement, ok := c.entries[key]
	if ok {
		c.evictionList.MoveToFront(oldElement)
		oldElement.Value.(*cacheEntry).value = value
		oldElement.Value.(*cacheEntry).expireTime = c.clock.Now().Add(ttl)
		return
	}

	// Make space if necessary
	if c.evictionList.Len() >= c.maxSize {
		toEvict := c.evictionList.Back()
		c.evictionList.Remove(toEvict)
		delete(c.entries, toEvict.Value.(*cacheEntry).key)
	}

	// Add new entry
	entry := &cacheEntry{
		key:        key,
		value:      value,
		expireTime: c.clock.Now().Add(ttl),
	}
	element := c.evictionList.PushFront(entry)
	c.entries[key] = element
}

// Get returns the value at the specified key from the cache if it exists and is not
// expired, or returns false.
func (c *LRUExpireCache) Get(key interface{}) (interface{}, bool) {
	c.lock.Lock()
	defer c.lock.Unlock()

	element, ok := c.entries[key]
	if !ok {
		return nil, false
	}

	if c.clock.Now().After(element.Value.(*cacheEntry).expireTime) {
		c.evictionList.Remove(element)
		delete(c.entries, key)
		return nil, false
	}

	c.evictionList.MoveToFront(element)

	return element.Value.(*cacheEntry).value, true
}

// Remove removes the specified key from the cache if it exists
func (c *LRUExpireCache) Remove(key interface{}) {
	c.lock.Lock()
	defer c.lock.Unlock()

	element, ok := c.entries[key]
	if !ok {
		return
	}

	c.evictionList.Remove(element)
	delete(c.entries, key)
}

// RemoveAll removes all keys that match predicate.
func (c *LRUExpireCache) RemoveAll(predicate func(key any) bool) {
	c.lock.Lock()
	defer c.lock.Unlock()

	for key, element := range c.entries {
		if predicate(key) {
			c.evictionList.Remove(element)
			delete(c.entries, key)
		}
	}
}

// Keys returns all unexpired keys in the cache.
//
// Keep in mind that subsequent calls to Get() for any of the returned keys
// might return "not found".
//
// Keys are returned ordered from least recently used to most recently used.
func (c *LRUExpireCache) Keys() []interface{} {
	c.lock.Lock()
	defer c.lock.Unlock()

	now := c.clock.Now()

	val := make([]interface{}, 0, c.evictionList.Len())
	for element := c.evictionList.Back(); element != nil; element = element.Prev() {
		// Only return unexpired keys
		if !now.After(element.Value.(*cacheEntry).expireTime) {
			val = append(val, element.Value.(*cacheEntry).key)
		}
	}

	return val
}
