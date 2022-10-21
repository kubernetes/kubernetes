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
	"github.com/kcp-dev/logicalcluster/v2"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/tools/cache"
)

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
	ByCluster(cluster logicalcluster.Name) cache.GenericLister
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

func (s *ClusterLister) ByCluster(cluster logicalcluster.Name) cache.GenericLister {
	return &genericLister{
		indexer:  s.indexer,
		resource: s.resource,
		cluster:  cluster,
	}
}

type genericLister struct {
	indexer  cache.Indexer
	cluster  logicalcluster.Name
	resource schema.GroupResource
}

func (s *genericLister) List(selector labels.Selector) (ret []runtime.Object, err error) {
	selectAll := selector == nil || selector.Empty()
	list, err := s.indexer.ByIndex(ClusterIndexName, ClusterIndexKey(s.cluster))
	if err != nil {
		return nil, err
	}

	for i := range list {
		item := list[i].(runtime.Object)
		if selectAll {
			ret = append(ret, item)
		} else {
			metadata, err := meta.Accessor(item)
			if err != nil {
				return nil, err
			}
			if selector.Matches(labels.Set(metadata.GetLabels())) {
				ret = append(ret, item)
			}
		}
	}

	return ret, err
}

func (s *genericLister) Get(name string) (runtime.Object, error) {
	key := ToClusterAwareKey(s.cluster.String(), "", name)
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
		cluster:   s.cluster,
	}
}

type genericNamespaceLister struct {
	indexer   cache.Indexer
	cluster   logicalcluster.Name
	namespace string
	resource  schema.GroupResource
}

func (s *genericNamespaceLister) List(selector labels.Selector) (ret []runtime.Object, err error) {
	selectAll := selector == nil || selector.Empty()

	list, err := s.indexer.ByIndex(ClusterAndNamespaceIndexName, ClusterAndNamespaceIndexKey(s.cluster, s.namespace))
	if err != nil {
		return nil, err
	}

	for i := range list {
		item := list[i].(runtime.Object)
		if selectAll {
			ret = append(ret, item)
		} else {
			metadata, err := meta.Accessor(item)
			if err != nil {
				return nil, err
			}
			if selector.Matches(labels.Set(metadata.GetLabels())) {
				ret = append(ret, item)
			}
		}
	}
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
