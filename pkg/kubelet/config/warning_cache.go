/*
Copyright The Kubernetes Authors.

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

package config

import (
	"sync"

	"k8s.io/utils/lru"
)

// warningCache is an LRU cache for deduplicating static pod warning messages.
type warningCache struct {
	cache *lru.Cache
	mu    sync.Mutex
}

// newWarningCache returns a warningCache.
func newWarningCache(maxCacheEntries int) *warningCache {
	return &warningCache{
		cache: lru.New(maxCacheEntries),
	}
}

// addIfAbsent atomically checks if a warning exists and adds it if not.
// Returns true if the warning was added (first occurrence), false if it already existed.
func (c *warningCache) addIfAbsent(warning string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if warning already exists
	if _, found := c.cache.Get(warning); found {
		return false
	}

	// Add the warning to the cache
	c.cache.Add(warning, nil)
	return true
}
