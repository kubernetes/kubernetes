/*
Copyright 2020 The Kubernetes Authors.

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

package kubelet

import (
	"sync"
	"time"

	"github.com/golang/groupcache/lru"

	"k8s.io/apimachinery/pkg/types"
)

// timeCache stores a time keyed by uid
type timeCache struct {
	lock  sync.Mutex
	cache *lru.Cache
}

// maxTimeCacheEntries is the cache entry number in lru cache. 1000 is a proper number
// for our 100 pods per node target. If we support more pods per node in the future, we
// may want to increase the number.
const maxTimeCacheEntries = 1000

func newTimeCache() *timeCache {
	return &timeCache{cache: lru.New(maxTimeCacheEntries)}
}

func (c *timeCache) Add(uid types.UID, t time.Time) {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.cache.Add(uid, t)
}

func (c *timeCache) Remove(uid types.UID) {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.cache.Remove(uid)
}

func (c *timeCache) Get(uid types.UID) (time.Time, bool) {
	c.lock.Lock()
	defer c.lock.Unlock()
	value, ok := c.cache.Get(uid)
	if !ok {
		return time.Time{}, false
	}
	t, ok := value.(time.Time)
	if !ok {
		return time.Time{}, false
	}
	return t, true
}
