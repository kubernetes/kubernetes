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
	"sync"
	"time"
)

type getFunc func(key string) (interface{}, error)
type listFunc func() (map[string]interface{}, error)

// timedCache is a cache with TTL.
type timedCache struct {
	store       map[string]interface{}
	lock        sync.Mutex
	ttl         time.Duration
	lastRefresh time.Time

	getter getFunc
	lister listFunc
}

// newTimedcache creates a new timedCache with TTL.
func newTimedcache(ttl time.Duration, getFunc getFunc, listFunc listFunc) *timedCache {
	return &timedCache{
		ttl:    ttl,
		getter: getFunc,
		lister: listFunc,
		store:  make(map[string]interface{}),
	}
}

// expired returns true if the cache is expired.
func (t *timedCache) expired() bool {
	return t.lastRefresh.Add(t.ttl).Before(time.Now())
}

// Get gets data by key. It returns nil if data not found.
func (t *timedCache) Get(key string) (interface{}, error) {
	t.lock.Lock()
	defer t.lock.Unlock()

	if t.expired() {
		if err := t.refresh(""); err != nil {
			return nil, err
		}
	}

	entry, ok := t.store[key]
	if ok {
		return entry, nil
	}

	// not cache this vm yet, refresh by key
	if err := t.refresh(key); err != nil {
		return nil, err
	}
	entry, ok = t.store[key]
	if ok {
		return entry, nil
	}

	// Key still not found, set it to nil. This is required to avoid
	// relisting nonexist objects.
	t.store[key] = nil
	return nil, nil
}

// refresh updates cache by getter or lister. If key is an empty string,
// then lister is used to update cache.
// It should be only called under a lock.
func (t *timedCache) refresh(key string) error {
	// Refresh cache by getter.
	if key != "" && t.getter != nil {
		data, err := t.getter(key)
		if err != nil {
			return err
		}

		t.store[key] = data
		return nil
	}

	// Refresh cache by lister.
	dataList, err := t.lister()
	if err != nil {
		return err
	}
	t.store = make(map[string]interface{})
	for key, data := range dataList {
		t.store[key] = data
	}

	t.lastRefresh = time.Now()
	return nil
}

// Delete removes data from cache by key.
func (t *timedCache) Delete(key string) {
	t.lock.Lock()
	defer t.lock.Unlock()

	// mark data as nil to avoid invoking APIs later.
	t.store[key] = nil
}

// List gets a list of data from cache.
func (t *timedCache) List() ([]interface{}, error) {
	t.lock.Lock()
	defer t.lock.Unlock()

	if t.expired() {
		if err := t.refresh(""); err != nil {
			return nil, err
		}
	}

	result := make([]interface{}, 0)
	for _, data := range t.store {
		if data == nil {
			continue
		}

		result = append(result, data)
	}
	return result, nil
}

// AddOrUpdate adds new data or updates existing data in the cache.
// If the data is nil, then it updates cache from getter.
func (t *timedCache) AddOrUpdate(key string, data interface{}) error {
	t.lock.Lock()
	defer t.lock.Unlock()

	if data != nil {
		t.store[key] = data
		return nil
	}

	return t.refresh(key)
}
