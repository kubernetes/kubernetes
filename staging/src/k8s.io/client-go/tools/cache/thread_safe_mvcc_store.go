/*
Copyright 2025 The Kubernetes Authors.

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
	"fmt"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/google/btree"
	"k8s.io/apimachinery/pkg/util/sets"
)

var _ ThreadSafeStore = (*threadSafeMVCCStore)(nil)

const mvccStoreDegree = 32

func newThreadSafeMVCCStore(indexers Indexers) ThreadSafeStore {
	s := &threadSafeMVCCStore{
		indexers: indexers,
	}
	snapshot := &mvccStoreSnapshot{
		data:    btree.NewG[mvccStoreKey[any]](mvccStoreDegree, isMVCCStoreKeyLess),
		indexes: make(map[string]*btree.BTreeG[mvccStoreKey[[]string]], len(indexers)),
	}
	for indexer := range indexers {
		snapshot.indexes[indexer] = btree.NewG[mvccStoreKey[[]string]](mvccStoreDegree, isMVCCStoreKeyLess)
	}
	s.snapshot.Store(snapshot)

	return s
}

type threadSafeMVCCStore struct {
	snapshot atomic.Pointer[mvccStoreSnapshot]

	// updateLock needs to be acquired when updating the store
	// to avoid dirty writes as well as when accessing the
	// indexers.
	updateLock sync.Mutex
	indexers   Indexers
}

type mvccStoreSnapshot struct {
	data    *btree.BTreeG[mvccStoreKey[any]]
	indexes map[string]*btree.BTreeG[mvccStoreKey[[]string]]
}

type mvccStoreKey[t any] struct {
	key   string
	value t
}

func isMVCCStoreKeyLess[t any](a, b mvccStoreKey[t]) bool {
	return a.key < b.key
}

func (s *threadSafeMVCCStore) Add(key string, obj any) {
	s.Update(key, obj)
}

func (s *threadSafeMVCCStore) Update(key string, obj any) {
	s.updateLock.Lock()
	defer s.updateLock.Unlock()

	snap := s.snapshot.Load()
	newData := snap.data.Clone()
	previous, _ := newData.ReplaceOrInsert(mvccStoreKey[any]{
		key:   key,
		value: obj,
	})
	newIndexes := s.updateIndexesLocked(snap.indexes, key, previous.value, obj)

	s.snapshot.Store(&mvccStoreSnapshot{
		data:    newData,
		indexes: newIndexes,
	})
}

func (s *threadSafeMVCCStore) Delete(key string) {
	s.updateLock.Lock()
	defer s.updateLock.Unlock()

	snap := s.snapshot.Load()
	newData := snap.data.Clone()
	previous, existed := newData.Delete(mvccStoreKey[any]{key: key})
	newIndexes := snap.indexes
	if existed {
		newIndexes = s.updateIndexesLocked(snap.indexes, key, previous.value, nil)
	}

	s.snapshot.Store(&mvccStoreSnapshot{
		data:    newData,
		indexes: newIndexes,
	})
}

func (s *threadSafeMVCCStore) Get(key string) (any, bool) {
	snap := s.snapshot.Load()
	raw, exists := snap.data.Get(mvccStoreKey[any]{key: key})
	if !exists {
		return nil, false
	}
	return raw.value, true
}

func (s *threadSafeMVCCStore) List() []any {
	snap := s.snapshot.Load()
	result := make([]any, 0, snap.data.Len())
	snap.data.Ascend(func(item mvccStoreKey[any]) bool {
		result = append(result, item.value)
		return true
	})

	return result
}

func (s *threadSafeMVCCStore) ListKeys() []string {
	snap := s.snapshot.Load()
	result := make([]string, 0, snap.data.Len())
	snap.data.Ascend(func(item mvccStoreKey[any]) bool {
		result = append(result, item.key)
		return true
	})
	return result
}

func (s *threadSafeMVCCStore) Replace(items map[string]interface{}, resourceVersion string) {
	s.updateLock.Lock()
	defer s.updateLock.Unlock()

	snap := &mvccStoreSnapshot{data: btree.NewG[mvccStoreKey[any]](mvccStoreDegree, isMVCCStoreKeyLess)}
	for k, v := range items {
		snap.data.ReplaceOrInsert(mvccStoreKey[any]{key: k, value: v})
		snap.indexes = s.updateIndexesLocked(snap.indexes, k, nil, v)
	}

	s.snapshot.Store(snap)
}

func (s *threadSafeMVCCStore) Index(indexName string, obj interface{}) ([]interface{}, error) {
	s.updateLock.Lock()
	indexFunc := s.indexers[indexName]
	s.updateLock.Unlock()
	if indexFunc == nil {
		return nil, fmt.Errorf("Index with name %s does not exist", indexName)
	}

	indexedValues, err := indexFunc(obj)
	if err != nil {
		return nil, err
	}

	snap := s.snapshot.Load()
	if snap.data.Len() == 0 {
		return nil, nil
	}

	storeKeySet := sets.New[string]()
	for _, indexedValue := range indexedValues {
		val, exists := snap.indexes[indexName].Get(mvccStoreKey[[]string]{key: indexedValue})
		if exists {
			storeKeySet.Insert(val.value...)
		}
	}

	list := make([]interface{}, 0, storeKeySet.Len())
	for k := range storeKeySet {
		raw, _ := snap.data.Get(mvccStoreKey[any]{key: k})
		list = append(list, raw.value)
	}

	return list, nil
}

func (s *threadSafeMVCCStore) IndexKeys(indexName, indexedValue string) ([]string, error) {
	snap := s.snapshot.Load()
	index, exists := snap.indexes[indexName]
	if !exists {
		return nil, fmt.Errorf("Index with name %s does not exist", indexName)
	}

	vals, exist := index.Get(mvccStoreKey[[]string]{key: indexedValue})
	if exist {
		return vals.value, nil
	}

	return nil, nil
}

func (s *threadSafeMVCCStore) ListIndexFuncValues(indexName string) []string {
	snap := s.snapshot.Load()

	index, exists := snap.indexes[indexName]
	if !exists {
		return nil
	}

	names := make([]string, 0, index.Len())
	index.Ascend(func(item mvccStoreKey[[]string]) bool {
		names = append(names, item.key)
		return true
	})

	return names
}

func (s *threadSafeMVCCStore) ByIndex(indexName, indexedValue string) ([]any, error) {
	snap := s.snapshot.Load()
	index, exists := snap.indexes[indexName]
	if !exists {
		return nil, fmt.Errorf("Index with name %s does not exist", indexName)
	}

	keys, hasKeys := index.Get(mvccStoreKey[[]string]{key: indexedValue})
	if !hasKeys {
		return nil, nil
	}

	result := make([]any, 0, len(keys.value))
	for _, key := range keys.value {
		item, _ := snap.data.Get(mvccStoreKey[any]{key: key})
		result = append(result, item.value)
	}

	return result, nil
}

func (s *threadSafeMVCCStore) GetIndexers() Indexers {
	s.updateLock.Lock()
	defer s.updateLock.Unlock()

	return s.indexers
}

func (s *threadSafeMVCCStore) AddIndexers(newIndexers Indexers) error {
	s.updateLock.Lock()
	defer s.updateLock.Unlock()

	oldKeys := sets.StringKeySet(s.indexers)
	newKeys := sets.StringKeySet(newIndexers)

	if oldKeys.HasAny(newKeys.List()...) {
		return fmt.Errorf("indexer conflict: %v", oldKeys.Intersection(newKeys))
	}

	snap := s.snapshot.Load()
	newIndixes := make(map[string]*btree.BTreeG[mvccStoreKey[[]string]], len(newIndexers)+len(snap.indexes))
	for name, indexFunc := range newIndexers {
		s.indexers[name] = indexFunc
		index := btree.NewG[mvccStoreKey[[]string]](mvccStoreDegree, isMVCCStoreKeyLess)
		snap.data.Ascend(func(item mvccStoreKey[any]) bool {
			index = s.updateIndexLocked(name, index, item.key, nil, item.value)
			return true
		})
		newIndixes[name] = index
	}

	for name, index := range snap.indexes {
		newIndixes[name] = index
	}

	s.snapshot.Store(&mvccStoreSnapshot{
		data:    snap.data,
		indexes: newIndixes,
	})

	return nil
}

func (s *threadSafeMVCCStore) Resync() error { return nil }

func (s *threadSafeMVCCStore) updateIndexesLocked(
	current map[string]*btree.BTreeG[mvccStoreKey[[]string]],
	key string,
	oldObj any,
	newObj any,
) map[string]*btree.BTreeG[mvccStoreKey[[]string]] {
	result := make(map[string]*btree.BTreeG[mvccStoreKey[[]string]], len(s.indexers))
	for name := range s.indexers {
		current := current[name]
		if current == nil {
			current = btree.NewG[mvccStoreKey[[]string]](mvccStoreDegree, isMVCCStoreKeyLess)
		}
		result[name] = s.updateIndexLocked(name, current, key, oldObj, newObj)
	}

	return result
}

func (s *threadSafeMVCCStore) updateIndexLocked(
	name string,
	current *btree.BTreeG[mvccStoreKey[[]string]],
	key string,
	oldObj any,
	newObj any,
) *btree.BTreeG[mvccStoreKey[[]string]] {
	indexFunc, ok := s.indexers[name]
	if !ok {
		// Should never happen. Caller is responsible for ensuring this exists, and should call with lock
		// held to avoid any races.
		panic(fmt.Errorf("indexer %q does not exist", name))
	}
	var oldValues, newValues []string
	var err error
	if oldObj != nil {
		oldValues, err = indexFunc(oldObj)
		if err != nil {
			panic(fmt.Errorf("unable to calculate an index entry for key %q on index %q: %w", key, name, err))
		}
	}
	if newObj != nil {
		newValues, err = indexFunc(newObj)
		if err != nil {
			panic(fmt.Errorf("unable to calculate an index entry for key %q on index %q: %w", key, name, err))
		}
	}

	oldValuesSet := sets.New(oldValues...)
	newValuesSet := sets.New(newValues...)
	if oldValuesSet.Equal(newValuesSet) {
		return current
	}

	result := current.Clone()
	for oldIndexVal := range oldValuesSet.Difference(newValuesSet) {
		idx, _ := result.Get(mvccStoreKey[[]string]{key: oldIndexVal})
		if len(idx.value) == 1 {
			_, _ = result.Delete(idx)
			continue
		}
		newIdx := make([]string, 0, len(idx.value)-1)
		for _, v := range idx.value {
			if v != key {
				newIdx = append(newIdx, v)
			}
		}
		result.ReplaceOrInsert(mvccStoreKey[[]string]{key: oldIndexVal, value: newIdx})
	}
	for newIndexVal := range newValuesSet.Difference(oldValuesSet) {
		idx, exists := result.Get(mvccStoreKey[[]string]{key: newIndexVal})
		if !exists {
			newIdx := make([]string, 1)
			newIdx[0] = key
			idx = mvccStoreKey[[]string]{key: newIndexVal, value: newIdx}
		} else {
			pos, _ := slices.BinarySearch(idx.value, key)
			idx = mvccStoreKey[[]string]{
				key:   newIndexVal,
				value: slices.Insert(slices.Clone(idx.value), pos, key),
			}
		}
		result.ReplaceOrInsert(idx)
	}

	return result
}
