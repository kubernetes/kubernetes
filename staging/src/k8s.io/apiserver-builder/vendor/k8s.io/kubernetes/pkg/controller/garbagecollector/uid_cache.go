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
	"k8s.io/apimachinery/pkg/types"
)

// UIDCache is an LRU cache for uid.
type UIDCache struct {
	mutex sync.Mutex
	cache *lru.Cache
}

// NewUIDCache returns a UIDCache.
func NewUIDCache(maxCacheEntries int) *UIDCache {
	return &UIDCache{
		cache: lru.New(maxCacheEntries),
	}
}

// Add adds a uid to the cache.
func (c *UIDCache) Add(uid types.UID) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.cache.Add(uid, nil)
}

// Has returns if a uid is in the cache.
func (c *UIDCache) Has(uid types.UID) bool {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	_, found := c.cache.Get(uid)
	return found
}
