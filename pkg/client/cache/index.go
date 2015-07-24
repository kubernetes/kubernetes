/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// Indexer is a storage interface that lets you list objects using multiple indexing functions
type Indexer interface {
	Store
	// Retrieve list of objects that match on the named indexing function
	Index(indexName string, obj interface{}) ([]interface{}, error)
	// ListIndexFuncValues returns the list of generated values of an Index func
	ListIndexFuncValues(indexName string) []string
}

// IndexFunc knows how to provide an indexed value for an object.
type IndexFunc func(obj interface{}) (string, error)

// MetaNamespaceIndexFunc is a default index function that indexes based on an object's namespace
func MetaNamespaceIndexFunc(obj interface{}) (string, error) {
	meta, err := meta.Accessor(obj)
	if err != nil {
		return "", fmt.Errorf("object has no meta: %v", err)
	}
	return meta.Namespace(), nil
}

// Index maps the indexed value to a set of keys in the store that match on that value
type Index map[string]util.StringSet

// Indexers maps a name to a IndexFunc
type Indexers map[string]IndexFunc

// Indices maps a name to an Index
type Indices map[string]Index
