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
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	storageerr "k8s.io/kubernetes/pkg/api/errors/storage"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/registry/namespace"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// rest implements a RESTStorage for namespaces against etcd
type REST struct {
	*registry.Store
	status *registry.Store
}

// StatusREST implements the REST endpoint for changing the status of a namespace.
type StatusREST struct {
	store *registry.Store
}

// FinalizeREST implements the REST endpoint for finalizing a namespace.
type FinalizeREST struct {
	store *registry.Store
}

// NewREST returns a RESTStorage object that will work against namespaces.
func NewREST(opts generic.RESTOptions) (*REST, *StatusREST, *FinalizeREST) {
	prefix := "/" + opts.ResourcePrefix

	newListFunc := func() runtime.Object { return &api.NamespaceList{} }
	storageInterface, _ := opts.Decorator(
		opts.StorageConfig,
		cachesize.GetWatchCacheSizeByResource(cachesize.Namespaces),
		&api.Namespace{},
		prefix,
		namespace.Strategy,
		newListFunc,
		storage.NoTriggerPublisher,
	)

	store := &registry.Store{
		NewFunc:     func() runtime.Object { return &api.Namespace{} },
		NewListFunc: newListFunc,
		KeyRootFunc: func(ctx api.Context) string {
			return prefix
		},
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return registry.NoNamespaceKeyFunc(ctx, prefix, name)
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.Namespace).Name, nil
		},
		PredicateFunc:           namespace.MatchNamespace,
		QualifiedResource:       api.Resource("namespaces"),
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,

		CreateStrategy:      namespace.Strategy,
		UpdateStrategy:      namespace.Strategy,
		DeleteStrategy:      namespace.Strategy,
		ReturnDeletedObject: true,

		Storage: storageInterface,
	}

	statusStore := *store
	statusStore.UpdateStrategy = namespace.StatusStrategy

	finalizeStore := *store
	finalizeStore.UpdateStrategy = namespace.FinalizeStrategy

	return &REST{Store: store, status: &statusStore}, &StatusREST{store: &statusStore}, &FinalizeREST{store: &finalizeStore}
}

// Delete enforces life-cycle rules for namespace termination
func (r *REST) Delete(ctx api.Context, name string, options *api.DeleteOptions) (runtime.Object, error) {
	nsObj, err := r.Get(ctx, name)
	if err != nil {
		return nil, err
	}

	namespace := nsObj.(*api.Namespace)

	// Ensure we have a UID precondition
	if options == nil {
		options = api.NewDeleteOptions(0)
	}
	if options.Preconditions == nil {
		options.Preconditions = &api.Preconditions{}
	}
	if options.Preconditions.UID == nil {
		options.Preconditions.UID = &namespace.UID
	} else if *options.Preconditions.UID != namespace.UID {
		err = apierrors.NewConflict(
			api.Resource("namespaces"),
			name,
			fmt.Errorf("Precondition failed: UID in precondition: %v, UID in object meta: %v", *options.Preconditions.UID, namespace.UID),
		)
		return nil, err
	}

	// upon first request to delete, we switch the phase to start namespace termination
	// TODO: enhance graceful deletion's calls to DeleteStrategy to allow phase change and finalizer patterns
	if namespace.DeletionTimestamp.IsZero() {
		key, err := r.Store.KeyFunc(ctx, name)
		if err != nil {
			return nil, err
		}

		preconditions := storage.Preconditions{UID: options.Preconditions.UID}

		out := r.Store.NewFunc()
		err = r.Store.Storage.GuaranteedUpdate(
			ctx, key, out, false, &preconditions,
			storage.SimpleUpdate(func(existing runtime.Object) (runtime.Object, error) {
				existingNamespace, ok := existing.(*api.Namespace)
				if !ok {
					// wrong type
					return nil, fmt.Errorf("expected *api.Namespace, got %v", existing)
				}
				// Set the deletion timestamp if needed
				if existingNamespace.DeletionTimestamp.IsZero() {
					now := unversioned.Now()
					existingNamespace.DeletionTimestamp = &now
				}
				// Set the namespace phase to terminating, if needed
				if existingNamespace.Status.Phase != api.NamespaceTerminating {
					existingNamespace.Status.Phase = api.NamespaceTerminating
				}
				return existingNamespace, nil
			}),
		)

		if err != nil {
			err = storageerr.InterpretGetError(err, api.Resource("namespaces"), name)
			err = storageerr.InterpretUpdateError(err, api.Resource("namespaces"), name)
			if _, ok := err.(*apierrors.StatusError); !ok {
				err = apierrors.NewInternalError(err)
			}
			return nil, err
		}

		return out, nil
	}

	// prior to final deletion, we must ensure that finalizers is empty
	if len(namespace.Spec.Finalizers) != 0 {
		err = apierrors.NewConflict(api.Resource("namespaces"), namespace.Name, fmt.Errorf("The system is ensuring all content is removed from this namespace.  Upon completion, this namespace will automatically be purged by the system."))
		return nil, err
	}
	return r.Store.Delete(ctx, name, options)
}

func (r *StatusREST) New() runtime.Object {
	return r.store.New()
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx api.Context, name string) (runtime.Object, error) {
	return r.store.Get(ctx, name)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}

func (r *FinalizeREST) New() runtime.Object {
	return r.store.New()
}

// Update alters the status finalizers subset of an object.
func (r *FinalizeREST) Update(ctx api.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}
