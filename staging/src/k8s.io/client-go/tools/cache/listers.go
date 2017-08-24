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
	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// AppendFunc is used to add a matching item to whatever list the caller is using
type AppendFunc func(interface{})

func ListAll(store Store, selector labels.Selector, appendFn AppendFunc) error {
	listOptions := metav1.ListOptions{}
	listOptions.LabelSelector = selector.String()
	listOptions.IncludeUninitialized = false
	return ListAllWithOptions(store, listOptions, appendFn)
}

// ListAllWithOptions only respects the Selector and the IncludeUninitialized of the options.
// AppendFunc is used to add a matching item to whatever list the caller is using
func ListAllWithOptions(store Store, options metav1.ListOptions, appendFn AppendFunc) error {
	includeUninitialized := options.IncludeUninitialized
	selector, err := labels.Parse(options.LabelSelector)
	if err != nil {
		return err
	}
	for _, m := range store.List() {
		metadata, err := meta.Accessor(m)
		if err != nil {
			return err
		}
		if !selector.Matches(labels.Set(metadata.GetLabels())) {
			continue
		}
		if !includeUninitialized && metadata.GetInitializers() != nil {
			continue
		}
		appendFn(m)
	}
	return nil
}

func ListAllByNamespace(indexer Indexer, namespace string, selector labels.Selector, appendFn AppendFunc) error {
	listOptions := metav1.ListOptions{}
	listOptions.LabelSelector = selector.String()
	listOptions.IncludeUninitialized = false
	return ListAllByNamespaceWithOptions(indexer, namespace, listOptions, appendFn)
}

// ListAllByNamespaceWithOptions only respects the Selector and the IncludeUninitialized of the options.
func ListAllByNamespaceWithOptions(indexer Indexer, namespace string, options metav1.ListOptions, appendFn AppendFunc) error {
	includeUninitialized := options.IncludeUninitialized
	selector, err := labels.Parse(options.LabelSelector)
	if err != nil {
		return err
	}
	if namespace == metav1.NamespaceAll {
		for _, m := range indexer.List() {
			metadata, err := meta.Accessor(m)
			if err != nil {
				return err
			}
			if !selector.Matches(labels.Set(metadata.GetLabels())) {
				continue
			}
			if !includeUninitialized && metadata.GetInitializers() != nil {
				continue
			}
			appendFn(m)
		}
		return nil
	}

	items, err := indexer.Index(NamespaceIndex, &metav1.ObjectMeta{Namespace: namespace})
	if err != nil {
		// Ignore error; do slow search without index.
		glog.Warningf("can not retrieve list of objects using index : %v", err)
		for _, m := range indexer.List() {
			metadata, err := meta.Accessor(m)
			if err != nil {
				return err
			}
			if metadata.GetNamespace() != namespace || !selector.Matches(labels.Set(metadata.GetLabels())) {
				continue
			}
			if !includeUninitialized && metadata.GetInitializers() != nil {
				continue
			}
			appendFn(m)
		}
		return nil
	}
	for _, m := range items {
		metadata, err := meta.Accessor(m)
		if err != nil {
			return err
		}
		if !selector.Matches(labels.Set(metadata.GetLabels())) {
			continue
		}
		if !includeUninitialized && metadata.GetInitializers() != nil {
			continue
		}
		appendFn(m)
	}

	return nil
}

// GenericLister is a lister skin on a generic Indexer
type GenericLister interface {
	// List will return all objects across namespaces
	List(selector labels.Selector) (ret []runtime.Object, err error)
	// ListAllWithOptions only respects the Selector and the IncludeUninitialized of the options.
	// AppendFunc is used to add a matching item to whatever list the caller is using
	ListWithOptions(options metav1.ListOptions) (ret []runtime.Object, err error)
	// Get will attempt to retrieve assuming that name==key
	Get(name string) (runtime.Object, error)
	// ByNamespace will give you a GenericNamespaceLister for one namespace
	ByNamespace(namespace string) GenericNamespaceLister
}

// GenericNamespaceLister is a lister skin on a generic Indexer
type GenericNamespaceLister interface {
	// List will return all objects in this namespace
	List(selector labels.Selector) (ret []runtime.Object, err error)
	// ListWithOptions only respects the Selector and the IncludeUninitialized of the options.
	// AppendFunc is used to add a matching item to whatever list the caller is using
	ListWithOptions(options metav1.ListOptions) (ret []runtime.Object, err error)
	// Get will attempt to retrieve by namespace and name
	Get(name string) (runtime.Object, error)
}

func NewGenericLister(indexer Indexer, resource schema.GroupResource) GenericLister {
	return &genericLister{indexer: indexer, resource: resource}
}

type genericLister struct {
	indexer  Indexer
	resource schema.GroupResource
}

func (s *genericLister) List(selector labels.Selector) (ret []runtime.Object, err error) {
	err = ListAll(s.indexer, selector, func(m interface{}) {
		ret = append(ret, m.(runtime.Object))
	})
	return ret, err
}

func (s *genericLister) ListWithOptions(options metav1.ListOptions) (ret []runtime.Object, err error) {
	err = ListAllWithOptions(s.indexer, options, func(m interface{}) {
		ret = append(ret, m.(runtime.Object))
	})
	return ret, err
}

func (s *genericLister) ByNamespace(namespace string) GenericNamespaceLister {
	return &genericNamespaceLister{indexer: s.indexer, namespace: namespace, resource: s.resource}
}

func (s *genericLister) Get(name string) (runtime.Object, error) {
	obj, exists, err := s.indexer.GetByKey(name)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.NewNotFound(s.resource, name)
	}
	return obj.(runtime.Object), nil
}

type genericNamespaceLister struct {
	indexer   Indexer
	namespace string
	resource  schema.GroupResource
}

func (s *genericNamespaceLister) List(selector labels.Selector) (ret []runtime.Object, err error) {
	err = ListAllByNamespace(s.indexer, s.namespace, selector, func(m interface{}) {
		ret = append(ret, m.(runtime.Object))
	})
	return ret, err
}

func (s *genericNamespaceLister) ListWithOptions(options metav1.ListOptions) (ret []runtime.Object, err error) {
	err = ListAllByNamespaceWithOptions(s.indexer, s.namespace, options, func(m interface{}) {
		ret = append(ret, m.(runtime.Object))
	})
	return ret, err
}

func (s *genericNamespaceLister) Get(name string) (runtime.Object, error) {
	obj, exists, err := s.indexer.GetByKey(s.namespace + "/" + name)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.NewNotFound(s.resource, name)
	}
	return obj.(runtime.Object), nil
}
