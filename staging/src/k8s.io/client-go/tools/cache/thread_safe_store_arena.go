//go:build linux && amd64 && !race

/*
Copyright The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/util/sets"
	clientgofeaturegate "k8s.io/client-go/features"
	"k8s.io/client-go/tools/cache/bytecache"
)

// newArenaThreadSafeStoreOrNil returns the experimental bytecache-backed
// ThreadSafeStore when KUBE_BYTECACHE=1, and nil otherwise (callers fall
// back to the default threadSafeMap). See the bytecache package docs.
func newArenaThreadSafeStoreOrNil(indexers Indexers, indices Indices, opts ...ThreadSafeStoreOption) ThreadSafeStore {
	if !bytecache.Enabled() {
		return nil
	}
	// Options are typed against threadSafeMap; harvest them via a throwaway.
	tmp := &threadSafeMap{}
	for _, opt := range opts {
		opt(tmp)
	}
	metrics := tmp.metrics
	if metrics == nil {
		metrics = newStoreMetrics(InformerNameAndResource{}, noopInformerMetricsProvider{})
	}
	return &arenaThreadSafeMap{
		cache:   bytecache.New(),
		index:   &storeIndex{indexers: indexers, indices: indices},
		metrics: metrics,
	}
}

// arenaThreadSafeMap mirrors threadSafeMap's semantics over a bytecache.Cache.
// Objects returned by Get/List are materialized copies sharing the informer
// contract: treat them as read-only. Unlike threadSafeMap, Get is not
// guaranteed to return pointer-identical objects across calls.
type arenaThreadSafeMap struct {
	lock    sync.RWMutex
	cache   *bytecache.Cache
	index   *storeIndex
	rv      string
	metrics *storeMetrics
}

var _ ThreadSafeStore = &arenaThreadSafeMap{}
var _ ThreadSafeStoreWithTransaction = &arenaThreadSafeMap{}

func (c *arenaThreadSafeMap) Transaction(txns ...ThreadSafeStoreTransaction) {
	if len(txns) == 0 {
		return
	}
	finalObj := txns[len(txns)-1].Object
	rv, rvErr := rvFromObject(finalObj)
	rvInt, parseErr := parseRVForMetricsWithTruncation(rv)
	c.lock.Lock()
	defer c.lock.Unlock()
	for _, txn := range txns {
		switch txn.Type {
		case TransactionTypeAdd, TransactionTypeUpdate:
			c.updateLocked(txn.Key, txn.Object)
		case TransactionTypeDelete:
			c.deleteLocked(txn.Key)
		}
	}
	if rvErr == nil {
		c.rv = rv
		if parseErr == nil {
			c.metrics.storeResourceVersion.Set(float64(rvInt))
		}
	}
}

func (c *arenaThreadSafeMap) Add(key string, obj interface{}) {
	c.Update(key, obj)
}

func (c *arenaThreadSafeMap) Update(key string, obj interface{}) {
	rv, rvErr := rvFromObject(obj)
	rvInt, parseErr := parseRVForMetricsWithTruncation(rv)
	c.lock.Lock()
	defer c.lock.Unlock()
	c.updateLocked(key, obj)
	if rvErr == nil {
		c.rv = rv
		if parseErr == nil {
			c.metrics.storeResourceVersion.Set(float64(rvInt))
		}
	}
}

func (c *arenaThreadSafeMap) updateLocked(key string, obj interface{}) {
	var oldObject interface{}
	if len(c.index.indexers) > 0 {
		// Materialize the previous version so indices can be unwound; the
		// index funcs only look at values, so a copy is equivalent.
		oldObject, _ = c.cache.GetTransient(key)
	}
	c.cache.Set(key, obj)
	c.index.updateIndices(oldObject, obj, key)
}

func (c *arenaThreadSafeMap) Delete(key string) {
	c.DeleteWithObject(key, nil)
}

func (c *arenaThreadSafeMap) DeleteWithObject(key string, obj interface{}) {
	var rv string
	var rvInt int64
	var rvErr, parseErr error
	if obj != nil {
		rv, rvErr = rvFromObject(obj)
		rvInt, parseErr = parseRVForMetricsWithTruncation(rv)
	}
	c.lock.Lock()
	defer c.lock.Unlock()
	c.deleteLocked(key)
	if obj != nil && rvErr == nil {
		c.rv = rv
		if parseErr == nil {
			c.metrics.storeResourceVersion.Set(float64(rvInt))
		}
	}
}

func (c *arenaThreadSafeMap) deleteLocked(key string) {
	if obj, exists := c.cache.GetTransient(key); exists {
		c.index.updateIndices(obj, nil, key)
		c.cache.Delete(key)
	}
}

func (c *arenaThreadSafeMap) Get(key string) (interface{}, bool) {
	c.lock.RLock()
	defer c.lock.RUnlock()
	return c.cache.Get(key)
}

func (c *arenaThreadSafeMap) List() []interface{} {
	c.lock.RLock()
	defer c.lock.RUnlock()
	return c.cache.All()
}

func (c *arenaThreadSafeMap) ListKeys() []string {
	c.lock.RLock()
	defer c.lock.RUnlock()
	return c.cache.Keys()
}

func (c *arenaThreadSafeMap) Replace(items map[string]interface{}, resourceVersion string) {
	var rvInt int64
	var parseErr error
	if resourceVersion != "" {
		rvInt, parseErr = parseRVForMetricsWithTruncation(resourceVersion)
	}
	c.lock.Lock()
	defer c.lock.Unlock()
	c.cache.Replace(items)
	c.rv = resourceVersion
	if parseErr == nil {
		c.metrics.storeResourceVersion.Set(float64(rvInt))
	}
	c.index.reset()
	for key, item := range items {
		c.index.updateIndices(nil, item, key)
	}
}

func (c *arenaThreadSafeMap) Index(indexName string, obj interface{}) ([]interface{}, error) {
	c.lock.RLock()
	defer c.lock.RUnlock()
	storeKeySet, err := c.index.getKeysFromIndex(indexName, obj)
	if err != nil {
		return nil, err
	}
	list := make([]interface{}, 0, storeKeySet.Len())
	for storeKey := range storeKeySet {
		if item, ok := c.cache.Get(storeKey); ok {
			list = append(list, item)
		}
	}
	return list, nil
}

func (c *arenaThreadSafeMap) ByIndex(indexName, indexedValue string) ([]interface{}, error) {
	c.lock.RLock()
	defer c.lock.RUnlock()
	set, err := c.index.getKeysByIndex(indexName, indexedValue)
	if err != nil {
		return nil, err
	}
	list := make([]interface{}, 0, set.Len())
	for key := range set {
		if item, ok := c.cache.Get(key); ok {
			list = append(list, item)
		}
	}
	return list, nil
}

func (c *arenaThreadSafeMap) IndexKeys(indexName, indexedValue string) ([]string, error) {
	c.lock.RLock()
	defer c.lock.RUnlock()
	set, err := c.index.getKeysByIndex(indexName, indexedValue)
	if err != nil {
		return nil, err
	}
	return sets.List(set), nil
}

func (c *arenaThreadSafeMap) ListIndexFuncValues(indexName string) []string {
	c.lock.RLock()
	defer c.lock.RUnlock()
	return c.index.getIndexValues(indexName)
}

func (c *arenaThreadSafeMap) GetIndexers() Indexers {
	return c.index.indexers
}

func (c *arenaThreadSafeMap) AddIndexers(newIndexers Indexers) error {
	c.lock.Lock()
	defer c.lock.Unlock()
	if err := c.index.addIndexers(newIndexers); err != nil {
		return err
	}
	for _, key := range c.cache.Keys() {
		item, ok := c.cache.GetTransient(key)
		if !ok {
			continue
		}
		for name := range newIndexers {
			c.index.updateSingleIndex(name, nil, item, key)
		}
	}
	return nil
}

func (c *arenaThreadSafeMap) Resync() error { return nil }

func (c *arenaThreadSafeMap) LastStoreSyncResourceVersion() string {
	if !clientgofeaturegate.FeatureGates().Enabled(clientgofeaturegate.AtomicFIFO) {
		return ""
	}
	c.lock.RLock()
	defer c.lock.RUnlock()
	return c.rv
}

func (c *arenaThreadSafeMap) Bookmark(rv string) {
	var rvInt int64
	var parseErr error
	if rv != "" {
		rvInt, parseErr = parseRVForMetricsWithTruncation(rv)
	}
	c.lock.Lock()
	defer c.lock.Unlock()
	c.rv = rv
	if parseErr == nil {
		c.metrics.storeResourceVersion.Set(float64(rvInt))
	}
}
