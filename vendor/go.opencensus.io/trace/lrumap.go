// Copyright 2019, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package trace

import (
	"github.com/golang/groupcache/lru"
)

// A simple lru.Cache wrapper that tracks the keys of the current contents and
// the cumulative number of evicted items.
type lruMap struct {
	cacheKeys    map[lru.Key]bool
	cache        *lru.Cache
	droppedCount int
}

func newLruMap(size int) *lruMap {
	lm := &lruMap{
		cacheKeys:    make(map[lru.Key]bool),
		cache:        lru.New(size),
		droppedCount: 0,
	}
	lm.cache.OnEvicted = func(key lru.Key, value interface{}) {
		delete(lm.cacheKeys, key)
		lm.droppedCount++
	}
	return lm
}

func (lm lruMap) len() int {
	return lm.cache.Len()
}

func (lm lruMap) keys() []interface{} {
	keys := make([]interface{}, 0, len(lm.cacheKeys))
	for k := range lm.cacheKeys {
		keys = append(keys, k)
	}
	return keys
}

func (lm *lruMap) add(key, value interface{}) {
	lm.cacheKeys[lru.Key(key)] = true
	lm.cache.Add(lru.Key(key), value)
}

func (lm *lruMap) get(key interface{}) (interface{}, bool) {
	return lm.cache.Get(key)
}
