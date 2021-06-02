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

package garbagecollector

import (
	"sync"

	"github.com/golang/groupcache/lru"
)

// ReferenceCache is an LRU cache for uid.
type ReferenceCache struct {
	mutex sync.Mutex
	cache *lru.Cache
}

// NewReferenceCache returns a ReferenceCache.
func NewReferenceCache(maxCacheEntries int) *ReferenceCache {
	return &ReferenceCache{
		cache: lru.New(maxCacheEntries),
	}
}

// Add adds a uid to the cache.
func (c *ReferenceCache) Add(reference objectReference) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.cache.Add(reference, nil)
}

// Has returns if a uid is in the cache.
func (c *ReferenceCache) Has(reference objectReference) bool {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	_, found := c.cache.Get(reference)
	return found
}
