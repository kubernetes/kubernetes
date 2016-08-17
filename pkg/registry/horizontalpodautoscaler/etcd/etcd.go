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

package etcd

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/registry/horizontalpodautoscaler"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

type REST struct {
	*registry.Store
}

// NewREST returns a RESTStorage object that will work against horizontal pod autoscalers.
func NewREST(opts generic.RESTOptions) (*REST, *StatusREST) {
	prefix := "/" + opts.ResourcePrefix

	newListFunc := func() runtime.Object { return &autoscaling.HorizontalPodAutoscalerList{} }
	storageInterface := opts.Decorator(
		opts.StorageConfig,
		cachesize.GetWatchCacheSizeByResource(cachesize.HorizontalPodAutoscalers),
		&autoscaling.HorizontalPodAutoscaler{},
		prefix,
		horizontalpodautoscaler.Strategy,
		newListFunc,
		storage.NoTriggerPublisher,
	)

	store := &registry.Store{
		NewFunc: func() runtime.Object { return &autoscaling.HorizontalPodAutoscaler{} },
		// NewListFunc returns an object capable of storing results of an etcd list.
		NewListFunc: newListFunc,
		// Produces a path that etcd understands, to the root of the resource
		// by combining the namespace in the context with the given prefix
		KeyRootFunc: func(ctx api.Context) string {
			return registry.NamespaceKeyRootFunc(ctx, prefix)
		},
		// Produces a path that etcd understands, to the resource by combining
		// the namespace in the context with the given prefix
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return registry.NamespaceKeyFunc(ctx, prefix, name)
		},
		// Retrieve the name field of an autoscaler
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*autoscaling.HorizontalPodAutoscaler).Name, nil
		},
		// Used to match objects based on labels/fields for list
		PredicateFunc:           horizontalpodautoscaler.MatchAutoscaler,
		QualifiedResource:       autoscaling.Resource("horizontalpodautoscalers"),
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,

		// Used to validate autoscaler creation
		CreateStrategy: horizontalpodautoscaler.Strategy,

		// Used to validate autoscaler updates
		UpdateStrategy: horizontalpodautoscaler.Strategy,
		DeleteStrategy: horizontalpodautoscaler.Strategy,

		Storage: storageInterface,
	}
	statusStore := *store
	statusStore.UpdateStrategy = horizontalpodautoscaler.StatusStrategy
	return &REST{store}, &StatusREST{store: &statusStore}
}

// StatusREST implements the REST endpoint for changing the status of a daemonset
type StatusREST struct {
	store *registry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &autoscaling.HorizontalPodAutoscaler{}
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx api.Context, name string) (runtime.Object, error) {
	return r.store.Get(ctx, name)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}
