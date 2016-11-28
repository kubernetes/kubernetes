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
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/core/configmap"
	"k8s.io/kubernetes/pkg/registry/generic"
	genericregistry "k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// REST implements a RESTStorage for ConfigMap against etcd
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work with ConfigMap objects.
func NewREST(opts generic.RESTOptions) *REST {
	prefix := "/" + opts.ResourcePrefix

	newListFunc := func() runtime.Object { return &api.ConfigMapList{} }
	storageInterface, dFunc := opts.Decorator(
		opts.StorageConfig,
		cachesize.GetWatchCacheSizeByResource(cachesize.ConfigMaps),
		&api.ConfigMap{},
		prefix,
		configmap.Strategy,
		newListFunc,
		configmap.GetAttrs,
		storage.NoTriggerPublisher)

	store := &genericregistry.Store{
		NewFunc: func() runtime.Object {
			return &api.ConfigMap{}
		},

		// NewListFunc returns an object to store results of an etcd list.
		NewListFunc: newListFunc,

		// Produces a path that etcd understands, to the root of the resource
		// by combining the namespace in the context with the given prefix.
		KeyRootFunc: func(ctx api.Context) string {
			return genericregistry.NamespaceKeyRootFunc(ctx, prefix)
		},

		// Produces a path that etcd understands, to the resource by combining
		// the namespace in the context with the given prefix
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return genericregistry.NamespaceKeyFunc(ctx, prefix, name)
		},

		// Retrieves the name field of a ConfigMap object.
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.ConfigMap).Name, nil
		},

		// Matches objects based on labels/fields for list and watch
		PredicateFunc: configmap.MatchConfigMap,

		QualifiedResource: api.Resource("configmaps"),

		EnableGarbageCollection: opts.EnableGarbageCollection,
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,

		CreateStrategy: configmap.Strategy,
		UpdateStrategy: configmap.Strategy,
		DeleteStrategy: configmap.Strategy,

		Storage:     storageInterface,
		DestroyFunc: dFunc,
	}
	return &REST{store}
}
