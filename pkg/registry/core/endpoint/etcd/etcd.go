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
	"k8s.io/kubernetes/pkg/registry/core/endpoint"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

type REST struct {
	*registry.Store
}

// NewREST returns a RESTStorage object that will work against endpoints.
func NewREST(opts generic.RESTOptions) *REST {
	prefix := "/" + opts.ResourcePrefix

	newListFunc := func() runtime.Object { return &api.EndpointsList{} }
	storageInterface, dFunc := opts.Decorator(
		opts.StorageConfig,
		cachesize.GetWatchCacheSizeByResource(cachesize.Endpoints),
		&api.Endpoints{},
		prefix,
		endpoint.Strategy,
		newListFunc,
		storage.NoTriggerPublisher,
	)

	store := &registry.Store{
		NewFunc:     func() runtime.Object { return &api.Endpoints{} },
		NewListFunc: newListFunc,
		KeyRootFunc: func(ctx api.Context) string {
			return registry.NamespaceKeyRootFunc(ctx, prefix)
		},
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return registry.NamespaceKeyFunc(ctx, prefix, name)
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.Endpoints).Name, nil
		},
		PredicateFunc:           endpoint.MatchEndpoints,
		QualifiedResource:       api.Resource("endpoints"),
		EnableGarbageCollection: opts.EnableGarbageCollection,
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,

		CreateStrategy: endpoint.Strategy,
		UpdateStrategy: endpoint.Strategy,
		DeleteStrategy: endpoint.Strategy,

		Storage:     storageInterface,
		DestroyFunc: dFunc,
	}
	return &REST{store}
}
