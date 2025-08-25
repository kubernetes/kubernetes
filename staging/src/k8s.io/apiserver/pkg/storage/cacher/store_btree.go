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
	"fmt"
	"strings"
	"sync"

	"github.com/google/btree"
	"k8s.io/client-go/tools/cache"
)

// newThreadedBtreeStoreIndexer returns a storage for cacher by adding locking over the two 2 data structures:
// * btree based storage for efficient LIST operation on prefix
// * map based indexer for retrieving values by index.
// This separation is used to allow independent snapshotting those two storages in the future.
// Intention is to utilize btree for its cheap snapshots that don't require locking if don't mutate data.
func newThreadedBtreeStoreIndexer(indexers cache.Indexers, degree int) *threadedStoreIndexer {
	return &threadedStoreIndexer{
		store:   newBtreeStore(degree),
		indexer: newIndexer(indexers),
	}
}

type threadedStoreIndexer struct {
	lock    sync.RWMutex
	store   btreeStore
	indexer indexer
}

var _ orderedLister = (*threadedStoreIndexer)(nil)

func (si *threadedStoreIndexer) Count(prefix, continueKey string) (count int) {
	si.lock.RLock()
	defer si.lock.RUnlock()
	return si.store.Count(prefix, continueKey)
}

func (si *threadedStoreIndexer) Clone() orderedLister {
	si.lock.RLock()
	defer si.lock.RUnlock()
	return si.store.Clone()
}

func (si *threadedStoreIndexer) Add(obj interface{}) error {
	return si.addOrUpdate(obj)
}

func (si *threadedStoreIndexer) Update(obj interface{}) error {
	return si.addOrUpdate(obj)
}

func (si *threadedStoreIndexer) addOrUpdate(obj interface{}) error {
	if obj == nil {
		return fmt.Errorf("obj cannot be nil")
	}
	newElem, ok := obj.(*storeElement)
	if !ok {
		return fmt.Errorf("obj not a storeElement: %#v", obj)
	}
	si.lock.Lock()
	defer si.lock.Unlock()
	oldElem := si.store.addOrUpdateElem(newElem)
	return si.indexer.updateElem(newElem.Key, oldElem, newElem)
}

func (si *threadedStoreIndexer) Delete(obj interface{}) error {
	storeElem, ok := obj.(*storeElement)
	if !ok {
		return fmt.Errorf("obj not a storeElement: %#v", obj)
	}
	si.lock.Lock()
	defer si.lock.Unlock()
	oldObj, existed := si.store.deleteElem(storeElem)
	if !existed {
		return nil
	}
	return si.indexer.updateElem(storeElem.Key, oldObj, nil)
}

func (si *threadedStoreIndexer) List() []interface{} {
	si.lock.RLock()
	defer si.lock.RUnlock()
	return si.store.List()
}

func (si *threadedStoreIndexer) ListPrefix(prefix, continueKey string) []interface{} {
	si.lock.RLock()
	defer si.lock.RUnlock()
	return si.store.ListPrefix(prefix, continueKey)
}

func (si *threadedStoreIndexer) ListKeys() []string {
	si.lock.RLock()
	defer si.lock.RUnlock()
	return si.store.ListKeys()
}

func (si *threadedStoreIndexer) Get(obj interface{}) (item interface{}, exists bool, err error) {
	si.lock.RLock()
	defer si.lock.RUnlock()
	return si.store.Get(obj)
}

func (si *threadedStoreIndexer) GetByKey(key string) (item interface{}, exists bool, err error) {
	si.lock.RLock()
	defer si.lock.RUnlock()
	return si.store.GetByKey(key)
}

func (si *threadedStoreIndexer) Replace(objs []interface{}, resourceVersion string) error {
	si.lock.Lock()
	defer si.lock.Unlock()
	err := si.store.Replace(objs, resourceVersion)
	if err != nil {
		return err
	}
	return si.indexer.Replace(objs, resourceVersion)
}

func (si *threadedStoreIndexer) ByIndex(indexName, indexValue string) ([]interface{}, error) {
	si.lock.RLock()
	defer si.lock.RUnlock()
	return si.indexer.ByIndex(indexName, indexValue)
}

func newBtreeStore(degree int) btreeStore {
	return btreeStore{
		tree: btree.NewG(degree, func(a, b *storeElement) bool {
			return a.Key < b.Key
		}),
	}
}

type btreeStore struct {
	tree *btree.BTreeG[*storeElement]
}

func (s *btreeStore) Clone() orderedLister {
	return &btreeStore{
		tree: s.tree.Clone(),
	}
}

func (s *btreeStore) Add(obj interface{}) error {
	if obj == nil {
		return fmt.Errorf("obj cannot be nil")
	}
	storeElem, ok := obj.(*storeElement)
	if !ok {
		return fmt.Errorf("obj not a storeElement: %#v", obj)
	}
	s.addOrUpdateElem(storeElem)
	return nil
}

func (s *btreeStore) Update(obj interface{}) error {
	if obj == nil {
		return fmt.Errorf("obj cannot be nil")
	}
	storeElem, ok := obj.(*storeElement)
	if !ok {
		return fmt.Errorf("obj not a storeElement: %#v", obj)
	}
	s.addOrUpdateElem(storeElem)
	return nil
}

func (s *btreeStore) Delete(obj interface{}) error {
	if obj == nil {
		return fmt.Errorf("obj cannot be nil")
	}
	storeElem, ok := obj.(*storeElement)
	if !ok {
		return fmt.Errorf("obj not a storeElement: %#v", obj)
	}
	s.deleteElem(storeElem)
	return nil
}

func (s *btreeStore) deleteElem(storeElem *storeElement) (*storeElement, bool) {
	return s.tree.Delete(storeElem)
}

func (s *btreeStore) List() []interface{} {
	items := make([]interface{}, 0, s.tree.Len())
	s.tree.Ascend(func(item *storeElement) bool {
		items = append(items, item)
		return true
	})
	return items
}

func (s *btreeStore) ListKeys() []string {
	items := make([]string, 0, s.tree.Len())
	s.tree.Ascend(func(item *storeElement) bool {
		items = append(items, item.Key)
		return true
	})
	return items
}

func (s *btreeStore) Get(obj interface{}) (item interface{}, exists bool, err error) {
	storeElem, ok := obj.(*storeElement)
	if !ok {
		return nil, false, fmt.Errorf("obj is not a storeElement")
	}
	item, exists = s.tree.Get(storeElem)
	return item, exists, nil
}

func (s *btreeStore) GetByKey(key string) (item interface{}, exists bool, err error) {
	return s.getByKey(key)
}

func (s *btreeStore) Replace(objs []interface{}, _ string) error {
	s.tree.Clear(false)
	for _, obj := range objs {
		storeElem, ok := obj.(*storeElement)
		if !ok {
			return fmt.Errorf("obj not a storeElement: %#v", obj)
		}
		s.addOrUpdateElem(storeElem)
	}
	return nil
}

// addOrUpdateLocked assumes a lock is held and is used for Add
// and Update operations.
func (s *btreeStore) addOrUpdateElem(storeElem *storeElement) *storeElement {
	oldObj, _ := s.tree.ReplaceOrInsert(storeElem)
	return oldObj
}

func (s *btreeStore) getByKey(key string) (item interface{}, exists bool, err error) {
	keyElement := &storeElement{Key: key}
	item, exists = s.tree.Get(keyElement)
	return item, exists, nil
}

func (s *btreeStore) ListPrefix(prefix, continueKey string) []interface{} {
	if continueKey == "" {
		continueKey = prefix
	}
	var result []interface{}
	s.tree.AscendGreaterOrEqual(&storeElement{Key: continueKey}, func(item *storeElement) bool {
		if !strings.HasPrefix(item.Key, prefix) {
			return false
		}
		result = append(result, item)
		return true
	})
	return result
}

func (s *btreeStore) Count(prefix, continueKey string) (count int) {
	if continueKey == "" {
		continueKey = prefix
	}
	s.tree.AscendGreaterOrEqual(&storeElement{Key: continueKey}, func(item *storeElement) bool {
		if !strings.HasPrefix(item.Key, prefix) {
			return false
		}
		count++
		return true
	})
	return count
}

// newIndexer returns a indexer similar to storeIndex from client-go/tools/cache.
// TODO: Unify the indexer code with client-go/cache package.
// Major differences is type of values stored and their mutability:
// * Indexer in client-go stores object keys, that are not mutable.
// * Indexer in cacher stores whole objects, which is mutable.
// Indexer in client-go uses keys as it is used in conjunction with map[key]value
// allowing for fast value retrieval, while btree used in cacher would provide additional overhead.
// Difference in mutability of stored values is used for optimizing some operations in client-go Indexer.
func newIndexer(indexers cache.Indexers) indexer {
	return indexer{
		indices:  map[string]map[string]map[string]*storeElement{},
		indexers: indexers,
	}
}

type indexer struct {
	indices  map[string]map[string]map[string]*storeElement
	indexers cache.Indexers
}

func (i *indexer) ByIndex(indexName, indexValue string) ([]interface{}, error) {
	indexFunc := i.indexers[indexName]
	if indexFunc == nil {
		return nil, fmt.Errorf("index with name %s does not exist", indexName)
	}
	index := i.indices[indexName]
	set := index[indexValue]
	list := make([]interface{}, 0, len(set))
	for _, obj := range set {
		list = append(list, obj)
	}
	return list, nil
}

func (i *indexer) Replace(objs []interface{}, resourceVersion string) error {
	i.indices = map[string]map[string]map[string]*storeElement{}
	for _, obj := range objs {
		storeElem, ok := obj.(*storeElement)
		if !ok {
			return fmt.Errorf("obj not a storeElement: %#v", obj)
		}
		err := i.updateElem(storeElem.Key, nil, storeElem)
		if err != nil {
			return err
		}
	}
	return nil
}

func (i *indexer) updateElem(key string, oldObj, newObj *storeElement) (err error) {
	var oldIndexValues, indexValues []string
	for name, indexFunc := range i.indexers {
		if oldObj != nil {
			oldIndexValues, err = indexFunc(oldObj)
		} else {
			oldIndexValues = oldIndexValues[:0]
		}
		if err != nil {
			return fmt.Errorf("unable to calculate an index entry for key %q on index %q: %w", key, name, err)
		}
		if newObj != nil {
			indexValues, err = indexFunc(newObj)
		} else {
			indexValues = indexValues[:0]
		}
		if err != nil {
			return fmt.Errorf("unable to calculate an index entry for key %q on index %q: %w", key, name, err)
		}
		index := i.indices[name]
		if index == nil {
			index = map[string]map[string]*storeElement{}
			i.indices[name] = index
		}
		if len(indexValues) == 1 && len(oldIndexValues) == 1 && indexValues[0] == oldIndexValues[0] {
			// We optimize for the most common case where indexFunc returns a single value which has not been changed
			i.add(key, indexValues[0], newObj, index)
			continue
		}
		for _, value := range oldIndexValues {
			i.delete(key, value, index)
		}
		for _, value := range indexValues {
			i.add(key, value, newObj, index)
		}
	}
	return nil
}

func (i *indexer) add(key, value string, obj *storeElement, index map[string]map[string]*storeElement) {
	set := index[value]
	if set == nil {
		set = map[string]*storeElement{}
		index[value] = set
	}
	set[key] = obj
}

func (i *indexer) delete(key, value string, index map[string]map[string]*storeElement) {
	set := index[value]
	if set == nil {
		return
	}
	delete(set, key)
	// If we don's delete the set when zero, indices with high cardinality
	// short lived resources can cause memory to increase over time from
	// unused empty sets. See `kubernetes/kubernetes/issues/84959`.
	if len(set) == 0 {
		delete(index, value)
	}
}

// newStoreSnapshotter returns a storeSnapshotter that stores snapshots for
// serving read requests with exact resource versions (RV) and pagination.
//
// Snapshots are created by calling Clone method on orderedLister, which is
// expected to be fast and efficient thanks to usage of B-trees.
// B-trees can create a lazy copy of the tree structure, minimizing overhead.
//
// Assuming the watch cache observes all events and snapshots cache after each of them,
// requests for a specific resource version can be served by retrieving
// the snapshot with the greatest RV less than or equal to the requested RV.
// To make snapshot retrivial efficient we need an ordered data structure, such as tree.
//
// The initial implementation uses a B-tree to achieve the following performance characteristics (n - number of snapshots stored):
//   - `Add`: Adds a new snapshot.
//     Complexity: O(log n).
//     Executed for each watch event observed by the cache.
//   - `GetLessOrEqual`: Retrieves the snapshot with the greatest RV less than or equal to the requested RV.
//     Complexity: O(log n).
//     Executed for each LIST request with match=Exact or continuation.
//   - `RemoveLess`: Cleans up snapshots outside the watch history window.
//     Complexity: O(k log n), k - number of snapshots to remove, usually only one if watch capacity was not reduced.
//     Executed per watch event observed when the cache is full.
//   - `Reset`: Cleans up all snapshots.
//     Complexity: O(1).
//     Executed when the watch cache is reinitialized.
//
// Further optimization is possible by leveraging the property that adds always
// increase the maximum RV and deletes only increase the minimum RV.
// For example, a binary search on a cyclic buffer of (RV, snapshot)
// should reduce number of allocations and improve removal complexity.
// However, this solution is more complex and is deferred for future implementation.
//
// TODO: Rewrite to use a cyclic buffer
func newStoreSnapshotter() *storeSnapshotter {
	s := &storeSnapshotter{
		snapshots: btree.NewG[rvSnapshot](btreeDegree, func(a, b rvSnapshot) bool {
			return a.resourceVersion < b.resourceVersion
		}),
	}
	return s
}

var _ Snapshotter = (*storeSnapshotter)(nil)

type Snapshotter interface {
	Reset()
	GetLessOrEqual(rv uint64) (orderedLister, bool)
	Add(rv uint64, indexer orderedLister)
	RemoveLess(rv uint64)
	Len() int
}

type storeSnapshotter struct {
	mux       sync.RWMutex
	snapshots *btree.BTreeG[rvSnapshot]
}

type rvSnapshot struct {
	resourceVersion uint64
	snapshot        orderedLister
}

func (s *storeSnapshotter) Reset() {
	s.mux.Lock()
	defer s.mux.Unlock()
	s.snapshots.Clear(false)
}

func (s *storeSnapshotter) GetLessOrEqual(rv uint64) (orderedLister, bool) {
	s.mux.RLock()
	defer s.mux.RUnlock()

	var result *rvSnapshot
	s.snapshots.DescendLessOrEqual(rvSnapshot{resourceVersion: rv}, func(rvs rvSnapshot) bool {
		result = &rvs
		return false
	})
	if result == nil {
		return nil, false
	}
	return result.snapshot, true
}

func (s *storeSnapshotter) Add(rv uint64, indexer orderedLister) {
	s.mux.Lock()
	defer s.mux.Unlock()
	s.snapshots.ReplaceOrInsert(rvSnapshot{resourceVersion: rv, snapshot: indexer.Clone()})
}

func (s *storeSnapshotter) RemoveLess(rv uint64) {
	s.mux.Lock()
	defer s.mux.Unlock()
	for s.snapshots.Len() > 0 {
		oldest, ok := s.snapshots.Min()
		if !ok {
			break
		}
		if rv <= oldest.resourceVersion {
			break
		}
		s.snapshots.DeleteMin()
	}
}

func (s *storeSnapshotter) Len() int {
	s.mux.RLock()
	defer s.mux.RUnlock()

	return s.snapshots.Len()
}
