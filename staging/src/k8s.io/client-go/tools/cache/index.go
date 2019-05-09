/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/util/sets"
)

// Indexer is a storage interface that lets you list objects using multiple indexing functions.
// There are three sorts of strings here.
// One is a storage key, as defined in the Store interface.
// Another sort is a name of an IndexFunc.
// The third sort of string is an "indexed value", produced by an IndexFunc.
type Indexer interface {
	Store
	// Index returns the stored objects whose set of indexed values
	// intersects the set of indexed values of the given object, where
	// the indexed value sets are produced by the named IndexFunc
	Index(indexName string, obj interface{}) ([]interface{}, error)
	// IndexKeys returns the storage keys of the stored objects whose
	// set of indexed values from the named IndexFunc includes the
	// given indexed value
	IndexKeys(indexName, indexedValue string) ([]string, error)
	// ListIndexFuncValues returns the union of the indexed value
	// sets, produced by the named IndexFunc, of the objects in this
	// Store
	ListIndexFuncValues(indexName string) []string
	// ByIndex returns the stored objects whose set of indexed values
	// from the named IndexFunc includes the given indexed value
	ByIndex(indexName, indexedValue string) ([]interface{}, error)
	// GetIndexer return the indexers
	GetIndexers() Indexers

	// AddIndexers adds more indexers to this store.  If you call this after you already have data
	// in the store, the results are undefined.
	AddIndexers(newIndexers Indexers) error
}

// IndexFunc knows how to extract the set of indexed values for an object.
type IndexFunc func(obj interface{}) ([]string, error)

// IndexFuncToKeyFuncAdapter adapts an indexFunc to a keyFunc.  This is only useful if your index function returns
// unique values for every object.  This is conversion can create errors when more than one key is found.  You
// should prefer to make proper key and index functions.
func IndexFuncToKeyFuncAdapter(indexFunc IndexFunc) KeyFunc {
	return func(obj interface{}) (string, error) {
		indexKeys, err := indexFunc(obj)
		if err != nil {
			return "", err
		}
		if len(indexKeys) > 1 {
			return "", fmt.Errorf("too many keys: %v", indexKeys)
		}
		if len(indexKeys) == 0 {
			return "", fmt.Errorf("unexpected empty indexKeys")
		}
		return indexKeys[0], nil
	}
}

const (
	NamespaceIndex string = "namespace"
)

// MetaNamespaceIndexFunc is a default index function that indexes based on an object's namespace
func MetaNamespaceIndexFunc(obj interface{}) ([]string, error) {
	meta, err := meta.Accessor(obj)
	if err != nil {
		return []string{""}, fmt.Errorf("object has no meta: %v", err)
	}
	return []string{meta.GetNamespace()}, nil
}

// Index maps the indexed value to a set of keys in the store that match on that value
type Index map[string]sets.String

// Indexers maps a name to a IndexFunc
type Indexers map[string]IndexFunc

// Indices maps a name to an Index
type Indices map[string]Index
