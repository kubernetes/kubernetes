/*
Copyright 2022 The Kubernetes Authors.

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

package cacher

import (
	"container/heap"
	"fmt"
	"strings"
	"sync"

	"github.com/google/btree"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/cache"
)

type btreeIndexer interface {
	cache.Store
	ByIndex(indexName, indexValue string) ([]interface{}, error)
	Clone() btreeIndexer
	LimitPrefixRead(limit int, prefixKey, continueKey string) ([]interface{}, bool)
	Count(prefixKey, continueKey string) int
}

type btreeStore struct {
	lock     sync.RWMutex
	tree     *btree.BTree
	indices  cache.Indices
	indexers cache.Indexers
	keyFunc  cache.KeyFunc
}

func newBtreeStore(keyFunc cache.KeyFunc, indexers cache.Indexers, degree int) *btreeStore {
	return &btreeStore{
		tree:     btree.New(degree),
		indices:  cache.Indices{},
		indexers: indexers,
		keyFunc:  keyFunc,
	}
}

func (t *btreeStore) Add(obj interface{}) error {
	t.lock.Lock()
	defer t.lock.Unlock()

	return t.addOrUpdateLocked(obj)
}

func (t *btreeStore) Update(obj interface{}) error {
	t.lock.Lock()
	defer t.lock.Unlock()

	return t.addOrUpdateLocked(obj)
}

func (t *btreeStore) Delete(obj interface{}) error {
	t.lock.Lock()
	defer t.lock.Unlock()

	storeElem, ok := obj.(*storeElement)
	if !ok {
		return fmt.Errorf("obj not a storeElement: %#v", obj)
	}

	key, err := t.keyFunc(obj)
	if err != nil {
		return cache.KeyError{Obj: obj, Err: err}
	}
	err = t.updateIndicesLocked(obj, nil, key)
	if err != nil {
		return err
	}

	item := t.tree.Delete(storeElem)
	if item == nil {
		return fmt.Errorf("obj does not exist")
	}

	return nil
}

func (t *btreeStore) List() []interface{} {
	t.lock.RLock()
	defer t.lock.RUnlock()

	items := make([]interface{}, 0, t.tree.Len())
	t.tree.Ascend(func(i btree.Item) bool {
		items = append(items, i.(interface{}))
		return true
	})

	return items
}

func (t *btreeStore) ListKeys() []string {
	t.lock.RLock()
	defer t.lock.RUnlock()

	items := make([]string, 0, t.tree.Len())
	t.tree.Ascend(func(i btree.Item) bool {
		items = append(items, i.(*storeElement).Key)
		return true
	})

	return items
}

func (t *btreeStore) Get(obj interface{}) (item interface{}, exists bool, err error) {
	t.lock.RLock()
	defer t.lock.RUnlock()

	storeElem, ok := obj.(*storeElement)
	if !ok {
		return nil, false, fmt.Errorf("obj is not a storeElement")
	}
	item = t.tree.Get(storeElem)
	if item == nil {
		return nil, false, nil
	}

	return item, true, nil
}

func (t *btreeStore) GetByKey(key string) (item interface{}, exists bool, err error) {
	t.lock.RLock()
	defer t.lock.RUnlock()

	return t.getByKeyLocked(key)
}

func (t *btreeStore) Replace(objs []interface{}, _ string) error {
	t.lock.Lock()
	defer t.lock.Unlock()

	t.tree.Clear(false)
	for _, obj := range objs {
		err := t.addOrUpdateLocked(obj)
		if err != nil {
			return err
		}
	}

	return nil
}

func (t *btreeStore) Resync() error {
	// Nothing to do.
	return nil
}

func (t *btreeStore) Clone() btreeIndexer {
	t.lock.Lock()
	defer t.lock.Unlock()

	return &btreeStore{
		tree:     t.tree.Clone(),
		indices:  t.indices,
		indexers: t.indexers,
		keyFunc:  t.keyFunc,
	}
}

// addOrUpdateLocked assumes a lock is held and is used for Add
// and Update operations.
func (t *btreeStore) addOrUpdateLocked(obj interface{}) error {
	// A nil obj cannot be entered into the btree,
	// results in panic.
	if obj == nil {
		return fmt.Errorf("obj cannot be nil")
	}
	storeElem, ok := obj.(*storeElement)
	if !ok {
		return fmt.Errorf("obj not a storeElement: %#v", obj)
	}

	returned := t.tree.ReplaceOrInsert(storeElem)
	key, err := t.keyFunc(obj)
	if err != nil {
		return cache.KeyError{Obj: obj, Err: err}
	}
	if returned == nil {
		return t.updateIndicesLocked(nil, obj, key)
	}

	old := returned.(interface{})
	return t.updateIndicesLocked(old, storeElem, key)
}

func (t *btreeStore) getByKeyLocked(key string) (item interface{}, exists bool, err error) {
	t.tree.Ascend(func(i btree.Item) bool {
		if key == i.(*storeElement).Key {
			item = i
			exists = true
			return false
		}
		return true
	})

	return item, exists, nil
}

func (t *btreeStore) updateIndicesLocked(oldObj, newObj interface{}, key string) error {
	var oldIndexValues, indexValues []string
	var err error
	for name, indexFunc := range t.indexers {
		if oldObj != nil {
			oldIndexValues, err = indexFunc(oldObj)
		} else {
			oldIndexValues = oldIndexValues[:0]
		}
		if err != nil {
			return fmt.Errorf("unable to calculate an index entry for key %q on index %q: %v", key, name, err)
		}

		if newObj != nil {
			indexValues, err = indexFunc(newObj)
		} else {
			indexValues = indexValues[:0]
		}
		if err != nil {
			return fmt.Errorf("unable to calculate an index entry for key %q on index %q: %v", key, name, err)
		}

		index := t.indices[name]
		if index == nil {
			index = cache.Index{}
			t.indices[name] = index
		}

		if len(indexValues) == 1 && len(oldIndexValues) == 1 && indexValues[0] == oldIndexValues[0] {
			// We optimize for the most common case where indexFunc returns a single value which has not been changed
			continue
		}

		for _, value := range oldIndexValues {
			t.deleteKeyFromIndexLocked(key, value, index)
		}
		for _, value := range indexValues {
			t.addKeyToIndexLocked(key, value, index)
		}
	}

	return nil
}

func (c *btreeStore) addKeyToIndexLocked(key, value string, index cache.Index) {
	set := index[value]
	if set == nil {
		set = sets.String{}
		index[value] = set
	}
	set.Insert(key)
}

func (t *btreeStore) deleteKeyFromIndexLocked(key, value string, index cache.Index) {
	set := index[value]
	if set == nil {
		return
	}
	set.Delete(key)
	// If we don't delete the set when zero, indices with high cardinality
	// short lived resources can cause memory to increase over time from
	// unused empty sets. See `kubernetes/kubernetes/issues/84959`.
	if len(set) == 0 {
		delete(index, value)
	}
}

func (t *btreeStore) LimitPrefixRead(limit int, prefixKey, continueKey string) ([]interface{}, bool) {
	t.lock.RLock()
	defer t.lock.RUnlock()

	if len(continueKey) > 0 {
		return t.limitPrefixReadLocked(limit, continueKey, prefixKey)
	}

	return t.limitPrefixReadLocked(limit, prefixKey, prefixKey)
}

func (t *btreeStore) Count(prefixKey, continueKey string) int {
	t.lock.RLock()
	defer t.lock.RUnlock()

	if len(continueKey) > 0 {
		return t.countLocked(continueKey, prefixKey)
	}

	return t.countLocked(prefixKey, prefixKey)
}

func (t *btreeStore) limitPrefixReadLocked(limit int, pivotKey, prefixKey string) ([]interface{}, bool) {
	if limit < 0 {
		return nil, false
	}
	var result []interface{}
	var hasMore bool

	if limit == 0 {
		t.tree.AscendGreaterOrEqual(&storeElement{Key: pivotKey}, func(i btree.Item) bool {
			elementKey := i.(*storeElement).Key
			if !strings.HasPrefix(elementKey, prefixKey) {
				return false
			}
			result = append(result, i.(interface{}))
			return true
		})
		return result, hasMore
	}

	t.tree.AscendGreaterOrEqual(&storeElement{Key: pivotKey}, func(i btree.Item) bool {
		elementKey := i.(*storeElement).Key
		if !strings.HasPrefix(elementKey, prefixKey) {
			return false
		}
		// TODO: Might be worth to lookup one more item to provide more accurate HasMore.
		if len(result) < limit {
			result = append(result, i.(interface{}))
		} else {
			hasMore = true
			return false
		}
		return true
	})

	return result, hasMore
}

func (t *btreeStore) countLocked(pivotKey, prefixKey string) (count int) {
	t.tree.AscendGreaterOrEqual(&storeElement{Key: pivotKey}, func(i btree.Item) bool {
		elementKey := i.(*storeElement).Key
		if !strings.HasPrefix(elementKey, prefixKey) {
			return false
		}
		count++
		return true
	})

	return count
}

func (t *btreeStore) ByIndex(indexName, indexValue string) ([]interface{}, error) {
	t.lock.RLock()
	defer t.lock.RUnlock()

	indexFunc := t.indexers[indexName]
	if indexFunc == nil {
		return nil, fmt.Errorf("Index with name %s does not exist", indexName)
	}

	index := t.indices[indexName]

	set := index[indexValue]
	list := make([]interface{}, 0, set.Len())
	for key := range set {
		obj, exists, err := t.getByKeyLocked(key)
		if err != nil {
			return nil, err
		}
		if !exists {
			return nil, fmt.Errorf("key %s does not exist in store", key)
		}
		list = append(list, obj)
	}

	return list, nil
}

var _ btreeIndexer = (*btreeStore)(nil)

// continueCache caches roots of trees that were created as
// clones to serve LIST requests. When a continue request is
// meant to be served for a certain LIST request, we retrieve
// the tree that served the LIST request and serve the continue
// request from there.
//
// A tree is removed from this cache when the RV at which it was
// created is removed from the watchCache.
type continueCache struct {
	sync.RWMutex
	revisions minIntHeap
	cache     map[uint64]btreeIndexer
}

func newContinueCache() *continueCache {
	return &continueCache{
		cache: make(map[uint64]btreeIndexer)}
}

func (c *continueCache) Get(rv uint64) (btreeIndexer, bool) {
	c.RLock()
	defer c.RUnlock()
	indexer, ok := c.cache[rv]
	return indexer, ok
}

func (c *continueCache) Set(rv uint64, indexer btreeIndexer) {
	c.Lock()
	defer c.Unlock()
	if _, ok := c.cache[rv]; !ok {
		heap.Push(&c.revisions, rv)
	}
	c.cache[rv] = indexer.Clone()
}

func (c *continueCache) Cleanup(rv uint64) {
	c.Lock()
	defer c.Unlock()
	for len(c.revisions) > 0 && rv >= c.revisions[0] {
		delRV := c.revisions[0]
		if _, ok := c.cache[delRV]; ok {
			delete(c.cache, delRV)
		}
		heap.Pop(&c.revisions)
	}
}

type minIntHeap []uint64

func (h minIntHeap) Len() int {
	return len(h)
}

func (h minIntHeap) Less(i, j int) bool {
	return h[i] < h[j]
}

func (h minIntHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *minIntHeap) Push(val interface{}) {
	*h = append(*h, val.(uint64))
}

func (h *minIntHeap) Pop() interface{} {
	old := *h

	size := len(old)
	val := old[size-1]
	*h = old[:size-1]

	return val
}
