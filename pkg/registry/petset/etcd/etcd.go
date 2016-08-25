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
	appsapi "k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/registry/petset"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// rest implements a RESTStorage for replication controllers against etcd
type REST struct {
	*registry.Store
}

// NewREST returns a RESTStorage object that will work against replication controllers.
func NewREST(opts generic.RESTOptions) (*REST, *StatusREST) {
	prefix := "/" + opts.ResourcePrefix

	newListFunc := func() runtime.Object { return &appsapi.PetSetList{} }
	storageInterface, _ := opts.Decorator(
		opts.StorageConfig,
		cachesize.GetWatchCacheSizeByResource(cachesize.PetSet),
		&appsapi.PetSet{},
		prefix,
		petset.Strategy,
		newListFunc,
		storage.NoTriggerPublisher,
	)

	store := &registry.Store{
		NewFunc: func() runtime.Object { return &appsapi.PetSet{} },

		// NewListFunc returns an object capable of storing results of an etcd list.
		NewListFunc: newListFunc,
		// Produces a petSet that etcd understands, to the root of the resource
		// by combining the namespace in the context with the given prefix
		KeyRootFunc: func(ctx api.Context) string {
			return registry.NamespaceKeyRootFunc(ctx, prefix)
		},
		// Produces a petSet that etcd understands, to the resource by combining
		// the namespace in the context with the given prefix
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return registry.NamespaceKeyFunc(ctx, prefix, name)
		},
		// Retrieve the name field of a replication controller
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*appsapi.PetSet).Name, nil
		},
		// Used to match objects based on labels/fields for list and watch
		PredicateFunc:           petset.MatchPetSet,
		QualifiedResource:       appsapi.Resource("petsets"),
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,

		// Used to validate controller creation
		CreateStrategy: petset.Strategy,

		// Used to validate controller updates
		UpdateStrategy: petset.Strategy,
		DeleteStrategy: petset.Strategy,

		Storage: storageInterface,
	}
	statusStore := *store
	statusStore.UpdateStrategy = petset.StatusStrategy
	return &REST{store}, &StatusREST{store: &statusStore}
}

// StatusREST implements the REST endpoint for changing the status of an petSet
type StatusREST struct {
	store *registry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &appsapi.PetSet{}
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx api.Context, name string) (runtime.Object, error) {
	return r.store.Get(ctx, name)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}
