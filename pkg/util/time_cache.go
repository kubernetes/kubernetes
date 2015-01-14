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
	ttl      time.Duration

	inFlight     map[string]chan T
	inFlightLock sync.Mutex

	cache map[string]timeCacheEntry
	lock  sync.RWMutex
}

// NewTimeCache returns a cache which calls fill to fill its entries, and
// forgets entries after ttl has passed.
func NewTimeCache(clock Clock, ttl time.Duration, fill func(key string) T) TimeCache {
	return &timeCache{
		clock:    clock,
		fillFunc: fill,
		inFlight: map[string]chan T{},
		cache:    map[string]timeCacheEntry{},
		ttl:      ttl,
	}
}

// Get returns the value of key from the cache, if it is present
// and recent enough; otherwise, it blocks while it gets the value.
func (c *timeCache) Get(key string) T {
	if item, ok := c.get(key); ok {
		return item
	}

	// We need to fill the cache. Calling the function could be
	// expensive, so do it while unlocked.
	wait := c.fillOrWait(key)
	item := <-wait

	// Put it back in the channel in case there's multiple waiters
	// (this channel is non-blocking)
	wait <- item
	return item
}

// returns the item and true if it is found and not expired, otherwise nil and false.
// If this returns false, it has locked c.inFlightLock and it is caller's responsibility
// to unlock that.
func (c *timeCache) get(key string) (T, bool) {
	c.lock.RLock()
	defer c.lock.RUnlock()
	data, ok := c.cache[key]
	now := c.clock.Now()
	if !ok || now.Sub(data.lastUpdate) > c.ttl {
		// We must lock this while we hold c.lock-- otherwise, a writer could
		// write to c.cache and remove the channel from c.inFlight before we
		// manage to read c.inFlight.
		c.inFlightLock.Lock()
		return nil, false
	}
	return data.item, true
}

// c.inFlightLock MUST be locked before calling this. fillOrWait will unlock it.
func (c *timeCache) fillOrWait(key string) chan T {
	defer c.inFlightLock.Unlock()

	// Already a call in progress?
	if current, ok := c.inFlight[key]; ok {
		return current
	}

	// We are the first, so we have to make the call.
	result := make(chan T, 1) // non-blocking
	c.inFlight[key] = result
	go func() {
		// Make potentially slow call.
		// While this call is in flight, fillOrWait will
		// presumably exit.
		data := timeCacheEntry{
			item:       c.fillFunc(key),
			lastUpdate: c.clock.Now(),
		}
		result <- data.item

		// Store in cache
		c.lock.Lock()
		defer c.lock.Unlock()
		c.cache[key] = data

		// Remove in flight entry
		c.inFlightLock.Lock()
		defer c.inFlightLock.Unlock()
		delete(c.inFlight, key)
	}()
	return result
}
