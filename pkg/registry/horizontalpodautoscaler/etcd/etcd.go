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

package etcd

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/registry/horizontalpodautoscaler"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

type REST struct {
	*etcdgeneric.Etcd
}

// NewREST returns a RESTStorage object that will work against horizontal pod autoscalers.
func NewREST(s storage.Interface, storageDecorator generic.StorageDecorator) (*REST, *StatusREST) {
	prefix := "/horizontalpodautoscalers"

	newListFunc := func() runtime.Object { return &extensions.HorizontalPodAutoscalerList{} }
	storageInterface := storageDecorator(
		s, 100, &extensions.HorizontalPodAutoscaler{}, prefix, horizontalpodautoscaler.Strategy, newListFunc)

	store := &etcdgeneric.Etcd{
		NewFunc: func() runtime.Object { return &extensions.HorizontalPodAutoscaler{} },
		// NewListFunc returns an object capable of storing results of an etcd list.
		NewListFunc: newListFunc,
		// Produces a path that etcd understands, to the root of the resource
		// by combining the namespace in the context with the given prefix
		KeyRootFunc: func(ctx api.Context) string {
			return etcdgeneric.NamespaceKeyRootFunc(ctx, prefix)
		},
		// Produces a path that etcd understands, to the resource by combining
		// the namespace in the context with the given prefix
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return etcdgeneric.NamespaceKeyFunc(ctx, prefix, name)
		},
		// Retrieve the name field of an autoscaler
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*extensions.HorizontalPodAutoscaler).Name, nil
		},
		// Used to match objects based on labels/fields for list
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return horizontalpodautoscaler.MatchAutoscaler(label, field)
		},
		QualifiedResource: extensions.Resource("horizontalpodautoscalers"),

		// Used to validate autoscaler creation
		CreateStrategy: horizontalpodautoscaler.Strategy,

		// Used to validate autoscaler updates
		UpdateStrategy: horizontalpodautoscaler.Strategy,

		Storage: storageInterface,
	}
	statusStore := *store
	statusStore.UpdateStrategy = horizontalpodautoscaler.StatusStrategy
	return &REST{store}, &StatusREST{store: &statusStore}
}

// StatusREST implements the REST endpoint for changing the status of a daemonset
type StatusREST struct {
	store *etcdgeneric.Etcd
}

func (r *StatusREST) New() runtime.Object {
	return &extensions.HorizontalPodAutoscaler{}
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	return r.store.Update(ctx, obj)
}
