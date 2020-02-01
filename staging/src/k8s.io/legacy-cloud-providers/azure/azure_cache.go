/*
Copyright 2017 The Kubernetes Authors.

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

package azure

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/client-go/tools/cache"
)

// cacheReadType defines the read type for cache data
type cacheReadType int

const (
	// cacheReadTypeDefault returns data from cache if cache entry not expired
	// if cache entry expired, then it will refetch the data using getter
	// save the entry in cache and then return
	cacheReadTypeDefault cacheReadType = iota
	// cacheReadTypeUnsafe returns data from cache even if the cache entry is
	// active/expired. If entry doesn't exist in cache, then data is fetched
	// using getter, saved in cache and returned
	cacheReadTypeUnsafe
	// cacheReadTypeForceRefresh force refreshes the cache even if the cache entry
	// is not expired
	cacheReadTypeForceRefresh
)

// getFunc defines a getter function for timedCache.
type getFunc func(key string) (interface{}, error)

// cacheEntry is the internal structure stores inside TTLStore.
type cacheEntry struct {
	key  string
	data interface{}

	// The lock to ensure not updating same entry simultaneously.
	lock sync.Mutex
	// time when entry was fetched and created
	createdOn time.Time
}

// cacheKeyFunc defines the key function required in TTLStore.
func cacheKeyFunc(obj interface{}) (string, error) {
	return obj.(*cacheEntry).key, nil
}

// timedCache is a cache with TTL.
type timedCache struct {
	store  cache.Store
	lock   sync.Mutex
	getter getFunc
	ttl    time.Duration
}

// newTimedcache creates a new timedCache.
func newTimedcache(ttl time.Duration, getter getFunc) (*timedCache, error) {
	if getter == nil {
		return nil, fmt.Errorf("getter is not provided")
	}

	return &timedCache{
		getter: getter,
		// switch to using NewStore instead of NewTTLStore so that we can
		// reuse entries for calls that are fine with reading expired/stalled data.
		// with NewTTLStore, entries are not returned if they have already expired.
		store: cache.NewStore(cacheKeyFunc),
		ttl:   ttl,
	}, nil
}

// getInternal returns cacheEntry by key. If the key is not cached yet,
// it returns a cacheEntry with nil data.
func (t *timedCache) getInternal(key string) (*cacheEntry, error) {
	entry, exists, err := t.store.GetByKey(key)
	if err != nil {
		return nil, err
	}
	// if entry exists, return the entry
	if exists {
		return entry.(*cacheEntry), nil
	}

	// lock here to ensure if entry doesn't exist, we add a new entry
	// avoiding overwrites
	t.lock.Lock()
	defer t.lock.Unlock()

	// Still not found, add new entry with nil data.
	// Note the data will be filled later by getter.
	newEntry := &cacheEntry{
		key:  key,
		data: nil,
	}
	t.store.Add(newEntry)
	return newEntry, nil
}

// Get returns the requested item by key.
func (t *timedCache) Get(key string, crt cacheReadType) (interface{}, error) {
	entry, err := t.getInternal(key)
	if err != nil {
		return nil, err
	}

	entry.lock.Lock()
	defer entry.lock.Unlock()

	// entry exists and if cache is not force refreshed
	if entry.data != nil && crt != cacheReadTypeForceRefresh {
		// allow unsafe read, so return data even if expired
		if crt == cacheReadTypeUnsafe {
			return entry.data, nil
		}
		// if cached data is not expired, return cached data
		if crt == cacheReadTypeDefault && time.Since(entry.createdOn) < t.ttl {
			return entry.data, nil
		}
	}
	// Data is not cached yet, cache data is expired or requested force refresh
	// cache it by getter. entry is locked before getting to ensure concurrent
	// gets don't result in multiple ARM calls.
	data, err := t.getter(key)
	if err != nil {
		return nil, err
	}

	// set the data in cache and also set the last update time
	// to now as the data was recently fetched
	entry.data = data
	entry.createdOn = time.Now().UTC()

	return entry.data, nil
}

// Delete removes an item from the cache.
func (t *timedCache) Delete(key string) error {
	return t.store.Delete(&cacheEntry{
		key: key,
	})
}

// Set sets the data cache for the key.
// It is only used for testing.
func (t *timedCache) Set(key string, data interface{}) {
	t.store.Add(&cacheEntry{
		key:       key,
		data:      data,
		createdOn: time.Now().UTC(),
	})
}
