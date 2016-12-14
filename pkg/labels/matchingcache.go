/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package labels

import (
	"hash/adler32"
	"sync"

	"github.com/davecgh/go-spew/spew"
	"github.com/golang/groupcache/lru"
)

var cache MatchingCache

const defaultCacheSize = 4096

type MatchingCache interface {
	Add(labels Labels, selector Selector, match bool)
	Get(labels Labels, selector Selector) (bool, bool)
}

func init() {
	cache = NewMatchingCache(defaultCacheSize)
}

func keyFunc(labels Labels, selector Selector) uint64 {
	hasher := adler32.New()
	printer := spew.ConfigState{
		Indent:         " ",
		SortKeys:       true,
		DisableMethods: true,
		SpewKeys:       true,
	}
	printer.Fprintf(hasher, "%#v%#v", &labels, &selector)

	return uint64(hasher.Sum32())
}

// matchingCache save label and selector matching relationship
type matchingCache struct {
	mutex sync.RWMutex
	cache *lru.Cache
}

// NewMatchingCache return a NewMatchingCache, which save label and selector matching relationship.
func NewMatchingCache(maxCacheEntries int) *matchingCache {
	return &matchingCache{
		cache: lru.New(maxCacheEntries),
	}
}

// Add will add matching information to the cache.
func (c *matchingCache) Add(labels Labels, selector Selector, match bool) {
	key := keyFunc(labels, selector)
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.cache.Add(key, match)
}

// Get returns the caching relationship of label and selector
func (c *matchingCache) Get(labels Labels, selector Selector) (bool, bool) {
	key := keyFunc(labels, selector)
	// NOTE: we use WLock instead of RLock here because lru's Get method is not threadsafe
	c.mutex.Lock()
	defer c.mutex.Unlock()
	match, hit := c.cache.Get(key)
	if hit {
		return match.(bool), hit
	}
	return false, false
}
