/*
Copyright 2021 The Kubernetes Authors.

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
package lru

import (
	"sync"

	groupcache "k8s.io/utils/internal/third_party/forked/golang/golang-lru"
)

type Key = groupcache.Key

// Cache is a thread-safe fixed size LRU cache.
type Cache struct {
	cache *groupcache.Cache
	lock  sync.RWMutex
}

// New creates an LRU of the given size.
func New(size int) *Cache {
	return &Cache{
		cache: groupcache.New(size),
	}
}

// Add adds a value to the cache.
func (c *Cache) Add(key Key, value interface{}) {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.cache.Add(key, value)
}

// Get looks up a key's value from the cache.
func (c *Cache) Get(key Key) (value interface{}, ok bool) {
	c.lock.RLock()
	defer c.lock.RUnlock()
	return c.cache.Get(key)
}

// Remove removes the provided key from the cache.
func (c *Cache) Remove(key Key) {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.cache.Remove(key)
}

// RemoveOldest removes the oldest item from the cache.
func (c *Cache) RemoveOldest() {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.cache.RemoveOldest()
}

// Len returns the number of items in the cache.
func (c *Cache) Len() int {
	c.lock.RLock()
	defer c.lock.RUnlock()
	return c.cache.Len()
}

// Clear purges all stored items from the cache.
func (c *Cache) Clear() {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.cache.Clear()
}
