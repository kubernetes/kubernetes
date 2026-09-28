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

	"k8s.io/client-go/tools/cache"
)

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
		indices:  map[string]map[string]map[string]*Element{},
		indexers: indexers,
	}
}

type indexer struct {
	indices  map[string]map[string]map[string]*Element
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

func (i *indexer) Replace(objs []*Element) error {
	i.indices = map[string]map[string]map[string]*Element{}
	for _, storeElem := range objs {
		err := i.updateElem(storeElem.Key, nil, storeElem)
		if err != nil {
			return err
		}
	}
	return nil
}

func (i *indexer) updateElem(key string, oldObj, newObj *Element) (err error) {
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
			index = map[string]map[string]*Element{}
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

func (i *indexer) add(key, value string, obj *Element, index map[string]map[string]*Element) {
	set := index[value]
	if set == nil {
		set = map[string]*Element{}
		index[value] = set
	}
	set[key] = obj
}

func (i *indexer) delete(key, value string, index map[string]map[string]*Element) {
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
