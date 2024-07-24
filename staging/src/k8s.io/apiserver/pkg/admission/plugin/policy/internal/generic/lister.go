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

package generic

import (
	"fmt"
	"net/http"

	kerrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/cache"

	kcpcache "github.com/kcp-dev/apimachinery/v2/pkg/cache"
	"github.com/kcp-dev/logicalcluster/v3"
)

var _ Lister[runtime.Object] = lister[runtime.Object]{}

type namespacedLister[T runtime.Object] struct {
	indexer     cache.Indexer
	namespace   string
	clusterName logicalcluster.Name
}

func (w namespacedLister[T]) List(selector labels.Selector) (ret []T, err error) {
	err = kcpcache.ListAllByClusterAndNamespace(w.indexer, w.clusterName, w.namespace, selector, func(m interface{}) {
		ret = append(ret, m.(T))
	})
	return ret, err
}

func (w namespacedLister[T]) Get(name string) (T, error) {
	var result T

	key := kcpcache.ToClusterAwareKey(w.clusterName.String(), w.namespace, name)

	obj, exists, err := w.indexer.GetByKey(key)
	if err != nil {
		return result, err
	}
	if !exists {
		return result, &kerrors.StatusError{ErrStatus: metav1.Status{
			Status:  metav1.StatusFailure,
			Code:    http.StatusNotFound,
			Reason:  metav1.StatusReasonNotFound,
			Message: fmt.Sprintf("%s not found", name),
		}}
	}
	result = obj.(T)
	return result, nil
}

type lister[T runtime.Object] struct {
	indexer     cache.Indexer
	clusterName logicalcluster.Name
}

func (w lister[T]) List(selector labels.Selector) (ret []T, err error) {
	err = kcpcache.ListAllByCluster(w.indexer, w.clusterName, selector, func(m interface{}) {
		ret = append(ret, m.(T))
	})
	return ret, err
}

func (w lister[T]) Get(name string) (T, error) {
	var result T

	key := kcpcache.ToClusterAwareKey(w.clusterName.String(), "", name)

	obj, exists, err := w.indexer.GetByKey(key)
	if err != nil {
		return result, err
	}
	if !exists {
		// kerrors.StatusNotFound requires a GroupResource we cannot provide
		return result, &kerrors.StatusError{ErrStatus: metav1.Status{
			Status:  metav1.StatusFailure,
			Code:    http.StatusNotFound,
			Reason:  metav1.StatusReasonNotFound,
			Message: fmt.Sprintf("%s not found", name),
		}}
	}
	result = obj.(T)
	return result, nil
}

func (w lister[T]) Namespaced(namespace string) NamespacedLister[T] {
	return namespacedLister[T]{namespace: namespace, indexer: w.indexer}
}

func NewLister[T runtime.Object](indexer cache.Indexer, clusterName logicalcluster.Name) lister[T] {
	return lister[T]{indexer: indexer, clusterName: clusterName}
}
