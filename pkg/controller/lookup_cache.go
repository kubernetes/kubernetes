/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package controller

import (
	"hash/adler32"
	"sync"

	"github.com/golang/groupcache/lru"
	"k8s.io/kubernetes/pkg/api/meta"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
)

type objectWithMeta interface {
	meta.Object
}

// keyFunc returns the key of an object, which is used to look up in the cache for it's matching object.
// Since we match objects by namespace and Labels/Selector, so if two objects have the same namespace and labels,
// they will have the same key.
func keyFunc(obj objectWithMeta) uint64 {
	hash := adler32.New()
	hashutil.DeepHashObject(hash, &equivalenceLabelObj{
		namespace: obj.GetNamespace(),
		labels:    obj.GetLabels(),
	})
	return uint64(hash.Sum32())
}

type equivalenceLabelObj struct {
	namespace string
	labels    map[string]string
}

// MatchingCache save label and selector matching relationship
type MatchingCache struct {
	mutex sync.RWMutex
	cache *lru.Cache
}

// NewMatchingCache return a NewMatchingCache, which save label and selector matching relationship.
func NewMatchingCache(maxCacheEntries int) *MatchingCache {
	return &MatchingCache{
		cache: lru.New(maxCacheEntries),
	}
}

// Add will add matching information to the cache.
func (c *MatchingCache) Add(labelObj objectWithMeta, selectorObj objectWithMeta) {
	key := keyFunc(labelObj)
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.cache.Add(key, selectorObj)
}

// GetMatchingObject lookup the matching object for a given object.
// Note: the cache information may be invalid since the controller may be deleted or updated,
// we need check in the external request to ensure the cache data is not dirty.
func (c *MatchingCache) GetMatchingObject(labelObj objectWithMeta) (controller interface{}, exists bool) {
	key := keyFunc(labelObj)
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return c.cache.Get(key)
}

// Update update the cached matching information.
func (c *MatchingCache) Update(labelObj objectWithMeta, selectorObj objectWithMeta) {
	c.Add(labelObj, selectorObj)
}

// InvalidateAll invalidate the whole cache.
func (c *MatchingCache) InvalidateAll() {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.cache = lru.New(c.cache.MaxEntries)
}
