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

	"k8s.io/apimachinery/pkg/util/clock"

	"github.com/hashicorp/golang-lru"
)

// LRUExpireCache is a cache that ensures the mostly recently accessed keys are returned with
// a ttl beyond which keys are forcibly expired.
type LRUExpireCache struct {
	// clock is used to obtain the current time
	clock clock.Clock

	cache *lru.Cache
	lock  sync.Mutex
}

// NewLRUExpireCache creates an expiring cache with the given size
func NewLRUExpireCache(maxSize int) *LRUExpireCache {
	return NewLRUExpireCacheWithClock(maxSize, clock.RealClock{})
}

// NewLRUExpireCacheWithClock creates an expiring cache with the given size, using the specified clock to obtain the current time.
func NewLRUExpireCacheWithClock(maxSize int, clock clock.Clock) *LRUExpireCache {
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

	// If not nil, a computation is in progress. You must wait and then
	// check finalValue.
	wait              <-chan struct{}
	finalValue        interface{}
	finalValueOmitted bool
}

// Add adds the value to the cache at key with the specified maximum duration.
func (c *LRUExpireCache) Add(key interface{}, value interface{}, ttl time.Duration) {
	c.lock.Lock()
	defer c.lock.Unlock()
	et := c.clock.Now().Add(ttl)
	c.cache.Add(key, &cacheEntry{
		value:      value,
		expireTime: et,
	})
}

// Get returns the value at the specified key from the cache if it exists and
// is not expired, or returns false. If the key is not in the cache because it
// is still being computed, Get will return immediately as if the key were not
// present. To wait, use GetOrWait.
func (c *LRUExpireCache) Get(key interface{}) (interface{}, bool) {
	ce, _ := c.getOrBegin(key, 0)
	if ce == nil {
		return nil, false
	}
	if ce.wait != nil {
		return nil, false
	}
	return ce.value, true
}

// ComputeFunc is used by GetOrWait to perform some work. You *must* exit if
// the abort chan is closed/has a value. On failure, return a ttl of 0 to avoid
// being cached; this will cause all waiting on the function to observe a cache miss.
type ComputeFunc func(abort <-chan time.Time) (value interface{}, ttl time.Duration)

// GetOrWait returns the value at the specified key from the cache if it exists
// and is not expired. Otherwise, it calls the compute() function to produce
// the value; requests for the same key that arrive within computationTime will
// be blocked while the computation is in progress; for all such requests,
// compute() will only be called once.
//
// In the unlikely event the cache has so much churn that a freshly computed
// value can't be regotten, this will attempt to compute it again, but not more
// than 3 times, and not if more than `computationTime` passes overall.
func (c *LRUExpireCache) GetOrWait(key interface{}, compute ComputeFunc, computationTime time.Duration) (interface{}, bool) {
	thisTTL := computationTime
	if thisTTL < 0 {
		thisTTL = 0
	}
	ce, ch := c.getOrBegin(key, thisTTL)
	if ch != nil {
		// We are the first caller for this key, so begin the
		// computation.
		abort := c.clock.After(thisTTL)
		go func() {
			defer close(ch)
			value, ttl := compute(abort)
			if ttl > 0 {
				// write a new entry before closing the channel
				// so that new getters of this key can check
				// ce.value and not wait; existing waiters
				// won't look at finalValue until after we
				// close the channel, so there is no race.
				c.Add(key, value, ttl)
				ce.finalValue = value
			} else {
				c.Remove(key)
				ce.finalValueOmitted = true
			}
		}()
	}
	if ce == nil {
		// This can happen if the computationTime was <= 0, preventing
		// us from caching even the attempt to do work.
		return nil, false
	}
	if ce.wait == nil {
		// ce.wait is nil, so there is no need to wait for a
		// computation to finish.
		return ce.value, true
	}

	// We must wait for a computation to finish.
	<-ce.wait
	if ce.finalValueOmitted {
		return nil, false
	}
	return ce.finalValue, true
}

// If ttl <= 0, it will not begin anything and instead return (nil, nil).
func (c *LRUExpireCache) getOrBegin(key interface{}, ttl time.Duration) (*cacheEntry, chan<- struct{}) {
	c.lock.Lock()
	defer c.lock.Unlock()
	e, ok := c.cache.Get(key)
	if ok {
		n := c.clock.Now()
		et := e.(*cacheEntry).expireTime
		if n.After(et) {
			c.cache.Remove(key)
			ok = false
		}
	}
	if ok {
		return e.(*cacheEntry), nil
	}
	if ttl <= 0 {
		return nil, nil
	}
	ch := make(chan struct{}, 1)
	ce := &cacheEntry{
		value:      nil,
		expireTime: c.clock.Now().Add(ttl),
		wait:       ch,
	}
	c.cache.Add(key, ce)
	return ce, ch
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
