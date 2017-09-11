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
	"hash/fnv"
	"time"
)

// split cache lookups across N striped caches
type stripedCache struct {
	stripeCount uint32
	keyFunc     func(string) uint32
	caches      []cache
}

type keyFunc func(string) uint32
type newCacheFunc func() cache

func newStripedCache(stripeCount int, keyFunc keyFunc, newCacheFunc newCacheFunc) cache {
	caches := []cache{}
	for i := 0; i < stripeCount; i++ {
		caches = append(caches, newCacheFunc())
	}
	return &stripedCache{
		stripeCount: uint32(stripeCount),
		keyFunc:     keyFunc,
		caches:      caches,
	}
}

func (c *stripedCache) get(key string) (*cacheRecord, bool) {
	return c.caches[c.keyFunc(key)%c.stripeCount].get(key)
}
func (c *stripedCache) set(key string, value *cacheRecord, ttl time.Duration) {
	c.caches[c.keyFunc(key)%c.stripeCount].set(key, value, ttl)
}
func (c *stripedCache) remove(key string) {
	c.caches[c.keyFunc(key)%c.stripeCount].remove(key)
}

func fnvKeyFunc(key string) uint32 {
	f := fnv.New32()
	f.Write([]byte(key))
	return f.Sum32()
}
