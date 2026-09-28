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

package store

import (
	"fmt"
	"iter"
	"strings"

	"k8s.io/utils/third_party/forked/golang/btree"
)

func newBtreeStore(degree int) btreeStore {
	return btreeStore{
		tree: btree.New(degree, func(a, b *Element) bool {
			return a.Key < b.Key
		}),
	}
}

type btreeStore struct {
	tree *btree.BTree[*Element]
}

// Clone should not be called concurrently.
// Ref: https://github.com/kubernetes/kubernetes/blob/4a8f617f3ca/vendor/k8s.io/utils/third_party/forked/golang/btree/btree.go#L586-L588
func (s *btreeStore) Clone() Snapshot {
	return &btreeStore{
		tree: s.tree.Clone(),
	}
}

func (s *btreeStore) deleteElem(storeElem *Element) (*Element, bool) {
	return s.tree.Delete(storeElem)
}

func (s *btreeStore) List() []interface{} {
	items := make([]interface{}, 0, s.tree.Len())
	s.tree.Ascend(func(item *Element) bool {
		items = append(items, item)
		return true
	})
	return items
}

func (s *btreeStore) ListKeys() []string {
	items := make([]string, 0, s.tree.Len())
	s.tree.Ascend(func(item *Element) bool {
		items = append(items, item.Key)
		return true
	})
	return items
}

func (s *btreeStore) Get(obj interface{}) (item interface{}, exists bool, err error) {
	storeElem, ok := obj.(*Element)
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
		storeElem, ok := obj.(*Element)
		if !ok {
			return fmt.Errorf("obj not a storeElement: %#v", obj)
		}
		s.addOrUpdateElem(storeElem)
	}
	return nil
}

// addOrUpdateLocked assumes a lock is held and is used for Add
// and Update operations.
func (s *btreeStore) addOrUpdateElem(storeElem *Element) *Element {
	oldObj, _ := s.tree.ReplaceOrInsert(storeElem)
	return oldObj
}

func (s *btreeStore) getByKey(key string) (item interface{}, exists bool, err error) {
	keyElement := &Element{Key: key}
	item, exists = s.tree.Get(keyElement)
	return item, exists, nil
}

func (s *btreeStore) OrderedListPrefix(prefix, continueKey string) ([]interface{}, error) {
	if continueKey == "" {
		continueKey = prefix
	}
	var result []interface{}
	s.tree.AscendGreaterOrEqual(&Element{Key: continueKey}, func(item *Element) bool {
		if !strings.HasPrefix(item.Key, prefix) {
			return false
		}
		result = append(result, item)
		return true
	})
	return result, nil
}

func (s *btreeStore) RangePrefix(prefix, continueKey string) Range {
	return prefixRange{s, prefix, continueKey}
}

func (s *btreeStore) rangePrefix(prefix, continueKey string) iter.Seq2[*Element, error] {
	if continueKey == "" {
		continueKey = prefix
	}
	return func(yield func(*Element, error) bool) {
		s.tree.AscendGreaterOrEqual(&Element{Key: continueKey}, func(item *Element) bool {
			if !strings.HasPrefix(item.Key, prefix) {
				return false
			}
			return yield(item, nil)
		})
	}
}

func (s *btreeStore) countPrefix(prefix, continueKey string) (count int) {
	if continueKey == "" {
		continueKey = prefix
	}
	s.tree.AscendGreaterOrEqual(&Element{Key: continueKey}, func(item *Element) bool {
		if !strings.HasPrefix(item.Key, prefix) {
			return false
		}
		count++
		return true
	})
	return count
}
