/*
Copyright 2014 Google Inc. All rights reserved.

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

package util

import (
	"sync"
	"time"
)

// T stands in for any type in TimeCache
// Should make it easy to use this as a template for an autogenerator
// if we ever start doing that.
type T interface{}

type TimeCache interface {
	// Get will fetch an item from the cache if
	// it is present and recent enough.
	Get(key string) T
}

type timeCacheEntry struct {
	item       T
	lastUpdate time.Time
}

type timeCache struct {
	clock    Clock
	fillFunc func(string) T
	cache    map[string]timeCacheEntry
	lock     sync.Mutex
	ttl      time.Duration
}

// NewTimeCache returns a cache which calls fill to fill its entries, and
// forgets entries after ttl has passed.
func NewTimeCache(clock Clock, ttl time.Duration, fill func(key string) T) TimeCache {
	return &timeCache{
		clock:    clock,
		fillFunc: fill,
		cache:    map[string]timeCacheEntry{},
		ttl:      ttl,
	}
}

// Get returns the value of key from the cache, if it is present
// and recent enough; otherwise, it blocks while it gets the value.
func (c *timeCache) Get(key string) T {
	c.lock.Lock()
	defer c.lock.Unlock()
	data, ok := c.cache[key]
	now := c.clock.Now()

	if !ok || now.Sub(data.lastUpdate) > c.ttl {
		data = timeCacheEntry{
			item:       c.fillFunc(key),
			lastUpdate: now,
		}
		c.cache[key] = data
	}
	return data.item
}
