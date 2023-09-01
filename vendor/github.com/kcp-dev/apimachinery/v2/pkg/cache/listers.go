/*
Copyright 2022 The KCP Authors.

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
	"github.com/kcp-dev/logicalcluster/v3"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

// ListAllByCluster used to list items belongs to a cluster from Indexer.
func ListAllByCluster(indexer cache.Indexer, clusterName logicalcluster.Name, selector labels.Selector, appendFn cache.AppendFunc) error {
	return listAllByIndexWithBackup(indexer, ClusterIndexName, ClusterIndexKey(clusterName), ClusterIndexFunc, selector, appendFn)
}

// ListAllByClusterAndNamespace used to list items belongs to a cluster and namespace from Indexer.
func ListAllByClusterAndNamespace(indexer cache.Indexer, clusterName logicalcluster.Name, namespace string, selector labels.Selector, appendFn cache.AppendFunc) error {
	if namespace == metav1.NamespaceAll {
		return ListAllByCluster(indexer, clusterName, selector, appendFn)
	}
	return listAllByIndexWithBackup(indexer, ClusterAndNamespaceIndexName, ClusterAndNamespaceIndexKey(clusterName, namespace), ClusterAndNamespaceIndexFunc, selector, appendFn)
}

// listAllByIndexWithBackup used to list items from the Indexer using an index, or falling back to a func if no index is registered.
func listAllByIndexWithBackup(indexer cache.Indexer, indexName, indexKey string, indexFunc cache.IndexFunc, selector labels.Selector, appendFn cache.AppendFunc) error {
	var items []interface{}
	var err error
	items, err = indexer.ByIndex(indexName, indexKey)
	if err != nil {
		// Ignore error; do slow search without index.
		klog.Warningf("can not retrieve list of objects using index : %v", err)
		for _, item := range indexer.List() {
			keys, err := indexFunc(item)
			if err != nil {
				return err
			}
			if sets.NewString(keys...).Has(indexKey) {
				items = append(items, item)
			}
		}
	}
	return appendMatchingObjects(items, selector, appendFn)
}

func appendMatchingObjects(items []interface{}, selector labels.Selector, appendFn cache.AppendFunc) error {
	selectAll := selector == nil || selector.Empty()
	for _, item := range items {
		if selectAll {
			// Avoid computing labels of the objects to speed up common flows
			// of listing all objects.
			appendFn(item)
			continue
		}
		metadata, err := meta.Accessor(item)
		if err != nil {
			return err
		}
		if selector.Matches(labels.Set(metadata.GetLabels())) {
			appendFn(item)
		}
	}

	return nil
}

// NewGenericClusterLister creates a new instance for the ClusterLister.
func NewGenericClusterLister(indexer cache.Indexer, resource schema.GroupResource) *ClusterLister {
	return &ClusterLister{
		indexer:  indexer,
		resource: resource,
	}
}

// GenericClusterLister is a lister that can either list all objects across all logical clusters, or
// scope down to a lister for one logical cluster only.
type GenericClusterLister interface {
	// List will return all objects across logical clusters and all namespaces
	List(selector labels.Selector) (ret []runtime.Object, err error)
	// ByCluster will give you a cache.GenericLister for one logical cluster
	ByCluster(clusterName logicalcluster.Name) cache.GenericLister
}

// ClusterLister is a lister that supports multiple logical clusters. It can list the entire contents of the backing store, and return individual cache.GenericListers that are scoped to individual logical clusters.
type ClusterLister struct {
	indexer  cache.Indexer
	resource schema.GroupResource
}

func (s *ClusterLister) List(selector labels.Selector) (ret []runtime.Object, err error) {
	if selector == nil {
		selector = labels.NewSelector()
	}
	err = cache.ListAll(s.indexer, selector, func(m interface{}) {
		ret = append(ret, m.(runtime.Object))
	})
	return ret, err
}

func (s *ClusterLister) ByCluster(clusterName logicalcluster.Name) cache.GenericLister {
	return &genericLister{
		indexer:     s.indexer,
		resource:    s.resource,
		clusterName: clusterName,
	}
}

type genericLister struct {
	indexer     cache.Indexer
	clusterName logicalcluster.Name
	resource    schema.GroupResource
}

func (s *genericLister) List(selector labels.Selector) (ret []runtime.Object, err error) {
	err = ListAllByCluster(s.indexer, s.clusterName, selector, func(i interface{}) {
		ret = append(ret, i.(runtime.Object))
	})
	return ret, err
}

func (s *genericLister) Get(name string) (runtime.Object, error) {
	key := ToClusterAwareKey(s.clusterName.String(), "", name)
	obj, exists, err := s.indexer.GetByKey(key)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.NewNotFound(s.resource, name)
	}
	return obj.(runtime.Object), nil
}

func (s *genericLister) ByNamespace(namespace string) cache.GenericNamespaceLister {
	return &genericNamespaceLister{
		indexer:   s.indexer,
		namespace: namespace,
		resource:  s.resource,
		cluster:   s.clusterName,
	}
}

type genericNamespaceLister struct {
	indexer   cache.Indexer
	cluster   logicalcluster.Name
	namespace string
	resource  schema.GroupResource
}

func (s *genericNamespaceLister) List(selector labels.Selector) (ret []runtime.Object, err error) {
	err = ListAllByClusterAndNamespace(s.indexer, s.cluster, s.namespace, selector, func(i interface{}) {
		ret = append(ret, i.(runtime.Object))
	})
	return ret, err
}

func (s *genericNamespaceLister) Get(name string) (runtime.Object, error) {
	key := ToClusterAwareKey(s.cluster.String(), s.namespace, name)
	obj, exists, err := s.indexer.GetByKey(key)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.NewNotFound(s.resource, name)
	}
	return obj.(runtime.Object), nil
}
