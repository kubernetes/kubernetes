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

package cache

import (
	"time"

	lrucache "k8s.io/apimachinery/pkg/util/cache"
	"k8s.io/apimachinery/pkg/util/clock"
)

type simpleCache struct {
	lru *lrucache.LRUExpireCache
}

func newSimpleCache(size int, clock clock.Clock) cache {
	return &simpleCache{lru: lrucache.NewLRUExpireCacheWithClock(size, clock)}
}

func (c *simpleCache) get(key string) (*cacheRecord, bool) {
	record, ok := c.lru.Get(key)
	if !ok {
		return nil, false
	}
	value, ok := record.(*cacheRecord)
	return value, ok
}

func (c *simpleCache) set(key string, value *cacheRecord, ttl time.Duration) {
	c.lru.Add(key, value, ttl)
}

func (c *simpleCache) remove(key string) {
	c.lru.Remove(key)
}
