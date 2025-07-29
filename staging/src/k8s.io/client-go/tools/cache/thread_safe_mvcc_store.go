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

	"github.com/benbjohnson/immutable"
	"k8s.io/apimachinery/pkg/util/sets"
)

var _ ThreadSafeStore = (*threadSafeMVCCStore)(nil)

const mvccStoreDegree = 32

func newThreadSafeMVCCStore(indexers Indexers) ThreadSafeStore {
	s := &threadSafeMVCCStore{
		indexers: indexers,
	}
	snapshot := &mvccStoreSnapshot{
		data:    immutable.NewMap[string, any](nil),
		indexes: make(map[string]*immutable.Map[string, []string], len(indexers)),
	}
	for indexer := range indexers {
		snapshot.indexes[indexer] = &immutable.Map[string, []string]{}
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
	data    *immutable.Map[string, any]
	indexes map[string]*immutable.Map[string, []string]
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

	previous, _ := snap.data.Get(key)
	newData := snap.data.Set(key, obj)
	newIndexes := s.updateIndexesLocked(snap.indexes, key, previous, obj)

	s.snapshot.Store(&mvccStoreSnapshot{
		data:    newData,
		indexes: newIndexes,
	})
}

func (s *threadSafeMVCCStore) Delete(key string) {
	s.updateLock.Lock()
	defer s.updateLock.Unlock()

	snap := s.snapshot.Load()
	previous, existed := snap.data.Get(key)
	if !existed {
		return
	}
	newData := snap.data.Delete(key)
	newIndexes := s.updateIndexesLocked(snap.indexes, key, previous, nil)

	s.snapshot.Store(&mvccStoreSnapshot{
		data:    newData,
		indexes: newIndexes,
	})
}

func (s *threadSafeMVCCStore) Get(key string) (any, bool) {
	snap := s.snapshot.Load()
	return snap.data.Get(key)
}

func (s *threadSafeMVCCStore) List() []any {
	snap := s.snapshot.Load()
	result := make([]any, 0, snap.data.Len())
	itr := snap.data.Iterator()
	for !itr.Done() {
		_, v, _ := itr.Next()
		result = append(result, v)
	}

	return result
}

func (s *threadSafeMVCCStore) ListKeys() []string {
	snap := s.snapshot.Load()
	result := make([]string, 0, snap.data.Len())
	itr := snap.data.Iterator()
	for !itr.Done() {
		k, _, _ := itr.Next()
		result = append(result, k)
	}
	return result
}

func (s *threadSafeMVCCStore) Replace(items map[string]interface{}, resourceVersion string) {
	s.updateLock.Lock()
	defer s.updateLock.Unlock()

	snap := &mvccStoreSnapshot{data: immutable.NewMapOf(nil, items)}
	for k, v := range items {
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
		val, exists := snap.indexes[indexName].Get(indexedValue)
		if exists {
			storeKeySet.Insert(val...)
		}
	}

	list := make([]interface{}, 0, storeKeySet.Len())
	for k := range storeKeySet {
		value, _ := snap.data.Get(k)
		list = append(list, value)
	}

	return list, nil
}

func (s *threadSafeMVCCStore) IndexKeys(indexName, indexedValue string) ([]string, error) {
	snap := s.snapshot.Load()
	index, exists := snap.indexes[indexName]
	if !exists {
		return nil, fmt.Errorf("Index with name %s does not exist", indexName)
	}

	vals, exist := index.Get(indexedValue)
	if exist {
		return vals, nil
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
	itr := index.Iterator()
	for !itr.Done() {
		k, _, _ := itr.Next()
		names = append(names, k)
	}

	return names
}

func (s *threadSafeMVCCStore) ByIndex(indexName, indexedValue string) ([]any, error) {
	snap := s.snapshot.Load()
	index, exists := snap.indexes[indexName]
	if !exists {
		return nil, fmt.Errorf("Index with name %s does not exist", indexName)
	}

	keys, hasKeys := index.Get(indexedValue)
	if !hasKeys {
		return nil, nil
	}

	result := make([]any, 0, len(keys))
	for _, key := range keys {
		item, _ := snap.data.Get(key)
		result = append(result, item)
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
	newIndixes := make(map[string]*immutable.Map[string, []string], len(newIndexers)+len(snap.indexes))
	for name, indexFunc := range newIndexers {
		s.indexers[name] = indexFunc
		index := immutable.NewMap[string, []string](nil)
		itr := snap.data.Iterator()
		for !itr.Done() {
			k, v, _ := itr.Next()
			index = s.updateIndexLocked(name, index, k, nil, v)
		}
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
	current map[string]*immutable.Map[string, []string],
	key string,
	oldObj any,
	newObj any,
) map[string]*immutable.Map[string, []string] {
	result := make(map[string]*immutable.Map[string, []string], len(s.indexers))
	for name := range s.indexers {
		current := current[name]
		if current == nil {
			current = immutable.NewMap[string, []string](nil)
		}
		result[name] = s.updateIndexLocked(name, current, key, oldObj, newObj)
	}

	return result
}

func (s *threadSafeMVCCStore) updateIndexLocked(
	name string,
	current *immutable.Map[string, []string],
	key string,
	oldObj any,
	newObj any,
) *immutable.Map[string, []string] {
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

	result := current
	for oldIndexVal := range oldValuesSet.Difference(newValuesSet) {
		idx, _ := result.Get(oldIndexVal)
		if len(idx) == 1 {
			result = result.Delete(oldIndexVal)
			continue
		}
		newIdx := make([]string, 0, len(idx)-1)
		for _, v := range idx {
			if v != key {
				newIdx = append(newIdx, v)
			}
		}
		result = result.Set(oldIndexVal, newIdx)
	}
	for newIndexVal := range newValuesSet.Difference(oldValuesSet) {
		idx, exists := result.Get(newIndexVal)
		if !exists {
			idx = make([]string, 1)
			idx[0] = key
		} else {
			pos, _ := slices.BinarySearch(idx, key)
			idx = slices.Insert(slices.Clone(idx), pos, key)
		}
		result = result.Set(newIndexVal, idx)
	}

	return result
}
