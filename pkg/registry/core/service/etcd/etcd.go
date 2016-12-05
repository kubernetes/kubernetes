/*
Copyright 2015 The Kubernetes Authors.

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
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/core/service"
	"k8s.io/kubernetes/pkg/registry/generic"
	genericregistry "k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against services.
func NewREST(opts generic.RESTOptions) (*REST, *StatusREST) {
	prefix := "/" + opts.ResourcePrefix

	newListFunc := func() runtime.Object { return &api.ServiceList{} }
	storageInterface, dFunc := opts.Decorator(
		opts.StorageConfig,
		cachesize.GetWatchCacheSizeByResource(cachesize.Services),
		&api.Service{},
		prefix,
		service.Strategy,
		newListFunc,
		service.GetAttrs,
		storage.NoTriggerPublisher,
	)

	store := &genericregistry.Store{
		NewFunc:     func() runtime.Object { return &api.Service{} },
		NewListFunc: newListFunc,
		KeyRootFunc: func(ctx api.Context) string {
			return genericregistry.NamespaceKeyRootFunc(ctx, prefix)
		},
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return genericregistry.NamespaceKeyFunc(ctx, prefix, name)
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.Service).Name, nil
		},
		PredicateFunc:           service.MatchServices,
		QualifiedResource:       api.Resource("services"),
		EnableGarbageCollection: opts.EnableGarbageCollection,
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,

		CreateStrategy: service.Strategy,
		UpdateStrategy: service.Strategy,
		DeleteStrategy: service.Strategy,
		ExportStrategy: service.Strategy,

		Storage:     storageInterface,
		DestroyFunc: dFunc,
	}
	statusStore := *store
	statusStore.UpdateStrategy = service.StatusStrategy
	return &REST{store}, &StatusREST{store: &statusStore}
}

// StatusREST implements the REST endpoint for changing the status of a service.
type StatusREST struct {
	store *genericregistry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &api.Service{}
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx api.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}
