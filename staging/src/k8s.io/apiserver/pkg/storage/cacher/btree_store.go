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
	"sync"

	"github.com/google/btree"
	"k8s.io/client-go/tools/cache"
)

type btreeIndexer interface {
	cache.Indexer
	Clone() btreeIndexer
}

type btreeStore struct {
	lock sync.RWMutex
	tree *btree.BTree
}

func newBtreeStore(degree int) *btreeStore {
	return &btreeStore{
		tree: btree.New(degree),
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

	storeElem, ok := obj.(storeElement)
	if !ok {
		return fmt.Errorf("obj not a storeElement: %#v", obj)
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
		items = append(items, i.(storeElement).Key)
		return true
	})

	return items
}

func (t *btreeStore) Get(obj interface{}) (item interface{}, exists bool, err error) {
	t.lock.RLock()
	defer t.lock.RUnlock()

	storeElem, ok := obj.(storeElement)
	if !ok {
		return nil, false, fmt.Errorf("obj is not a storeElement")
	}
	item = t.tree.Get(storeElem)
	if item == nil {
		return nil, false, nil
	}

	return item, false, nil
}

func (t *btreeStore) GetByKey(key string) (item interface{}, exists bool, err error) {
	t.lock.RLock()
	defer t.lock.RUnlock()

	t.tree.Ascend(func(i btree.Item) bool {
		if key == i.(storeElement).Key {
			item = i
			exists = true
			return false
		}
		return true
	})

	return item, exists, nil
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

// TODO(MadhavJivrajani): This is un-implemented for now. Stubbing this out
// to implement cacher.Indexer
func (t *btreeStore) Index(indexName string, obj interface{}) ([]interface{}, error) {
	return nil, fmt.Errorf("yet to implement")
}

// TODO(MadhavJivrajani): This is un-implemented for now. Stubbing this out
// to implement cacher.Indexer
func (t *btreeStore) IndexKeys(indexName, indexedValue string) ([]string, error) {
	return nil, fmt.Errorf("yet to implement")
}

// TODO(MadhavJivrajani): This is un-implemented for now. Stubbing this out
// to implement cacher.Indexer
func (t *btreeStore) ListIndexFuncValues(indexName string) []string {
	return nil
}

// TODO(MadhavJivrajani): This is un-implemented for now. Stubbing this out
// to implement cacher.Indexer
func (t *btreeStore) ByIndex(indexName, indexedValue string) ([]interface{}, error) {
	return nil, fmt.Errorf("yet to implement")
}

// TODO(MadhavJivrajani): This is un-implemented for now. Stubbing this out
// to implement cacher.Indexer
func (t *btreeStore) GetIndexers() cache.Indexers {
	return nil
}

// TODO(MadhavJivrajani): This is un-implemented for now. Stubbing this out
// to implement cacher.Indexer
func (t *btreeStore) AddIndexers(newIndexers cache.Indexers) error {
	return fmt.Errorf("yet to implement")
}

func (t *btreeStore) Clone() btreeIndexer {
	t.lock.Lock()
	defer t.lock.Unlock()

	return &btreeStore{tree: t.tree.Clone()}
}

// addOrUpdateLocked assumes a lock is held and is used for Add
// and Update operations.
func (t *btreeStore) addOrUpdateLocked(obj interface{}) error {
	// A nil obj cannot be entered into the btree,
	// results in panic.
	if obj == nil {
		return fmt.Errorf("obj cannot be nil")
	}
	storeElem, ok := obj.(storeElement)
	if !ok {
		return fmt.Errorf("obj not a storeElement: %#v", obj)
	}
	t.tree.ReplaceOrInsert(storeElem)

	return nil
}

// TODO: The goal is to make an acutal btreeIndexer.
// For now, only cache.Store methods are implemented,
// and the remaining methods of cache.Indexer are only
// stubbed out.
var _ btreeIndexer = (*btreeStore)(nil)
