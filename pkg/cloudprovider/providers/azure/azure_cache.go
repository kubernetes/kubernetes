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

// getFunc defines a getter function for timedCache.
type getFunc func(key string) (interface{}, error)

// cacheEntry is the internal structure stores inside TTLStore.
type cacheEntry struct {
	key  string
	data interface{}

	// The lock to ensure not updating same entry simultaneously.
	lock sync.Mutex
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
}

// newTimedcache creates a new timedCache.
func newTimedcache(ttl time.Duration, getter getFunc) (*timedCache, error) {
	if getter == nil {
		return nil, fmt.Errorf("getter is not provided")
	}

	return &timedCache{
		getter: getter,
		store:  cache.NewTTLStore(cacheKeyFunc, ttl),
	}, nil
}

// getInternal returns cacheEntry by key. If the key is not cached yet,
// it returns a cacheEntry with nil data.
func (t *timedCache) getInternal(key string) (*cacheEntry, error) {
	entry, exists, err := t.store.GetByKey(key)
	if err != nil {
		return nil, err
	}
	if exists {
		return entry.(*cacheEntry), nil
	}

	t.lock.Lock()
	defer t.lock.Unlock()
	entry, exists, err = t.store.GetByKey(key)
	if err != nil {
		return nil, err
	}
	if exists {
		return entry.(*cacheEntry), nil
	}

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
func (t *timedCache) Get(key string) (interface{}, error) {
	entry, err := t.getInternal(key)
	if err != nil {
		return nil, err
	}

	// Data is still not cached yet, cache it by getter.
	if entry.data == nil {
		entry.lock.Lock()
		defer entry.lock.Unlock()

		if entry.data == nil {
			data, err := t.getter(key)
			if err != nil {
				return nil, err
			}

			entry.data = data
		}
	}

	return entry.data, nil
}

// Delete removes an item from the cache.
func (t *timedCache) Delete(key string) error {
	return t.store.Delete(&cacheEntry{
		key: key,
	})
}
