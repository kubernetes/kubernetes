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

	"k8s.io/client-go/tools/cache"
)

type timedcacheEntry struct {
	key  string
	data interface{}
}

type timedcache struct {
	store cache.Store
	lock  sync.Mutex
}

// ttl time.Duration
func newTimedcache(ttl time.Duration) timedcache {
	return timedcache{
		store: cache.NewTTLStore(cacheKeyFunc, ttl),
	}
}

func cacheKeyFunc(obj interface{}) (string, error) {
	return obj.(*timedcacheEntry).key, nil
}

func (t *timedcache) GetOrCreate(key string, createFunc func() interface{}) (interface{}, error) {
	entry, exists, err := t.store.GetByKey(key)
	if err != nil {
		return nil, err
	}
	if exists {
		return (entry.(*timedcacheEntry)).data, nil
	}

	t.lock.Lock()
	defer t.lock.Unlock()
	entry, exists, err = t.store.GetByKey(key)
	if err != nil {
		return nil, err
	}
	if exists {
		return (entry.(*timedcacheEntry)).data, nil
	}

	if createFunc == nil {
		return nil, nil
	}
	created := createFunc()
	t.store.Add(&timedcacheEntry{
		key:  key,
		data: created,
	})
	return created, nil
}

func (t *timedcache) Delete(key string) {
	_ = t.store.Delete(&timedcacheEntry{
		key: key,
	})
}
