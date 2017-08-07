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

package storage

import (
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	storageerr "k8s.io/apiserver/pkg/storage/errors"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/core/namespace"
)

// rest implements a RESTStorage for namespaces
type REST struct {
	store  *genericregistry.Store
	status *genericregistry.Store
}

// StatusREST implements the REST endpoint for changing the status of a namespace.
type StatusREST struct {
	store *genericregistry.Store
}

// FinalizeREST implements the REST endpoint for finalizing a namespace.
type FinalizeREST struct {
	store *genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against namespaces.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, *StatusREST, *FinalizeREST) {
	store := &genericregistry.Store{
		Copier:                   api.Scheme,
		NewFunc:                  func() runtime.Object { return &api.Namespace{} },
		NewListFunc:              func() runtime.Object { return &api.NamespaceList{} },
		PredicateFunc:            namespace.MatchNamespace,
		DefaultQualifiedResource: api.Resource("namespaces"),
		WatchCacheSize:           cachesize.GetWatchCacheSizeByResource("namespaces"),

		CreateStrategy:      namespace.Strategy,
		UpdateStrategy:      namespace.Strategy,
		DeleteStrategy:      namespace.Strategy,
		ReturnDeletedObject: true,
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: namespace.GetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}

	statusStore := *store
	statusStore.UpdateStrategy = namespace.StatusStrategy

	finalizeStore := *store
	finalizeStore.UpdateStrategy = namespace.FinalizeStrategy

	return &REST{store: store, status: &statusStore}, &StatusREST{store: &statusStore}, &FinalizeREST{store: &finalizeStore}
}

func (r *REST) New() runtime.Object {
	return r.store.New()
}

func (r *REST) NewList() runtime.Object {
	return r.store.NewList()
}

func (r *REST) List(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (runtime.Object, error) {
	return r.store.List(ctx, options)
}

func (r *REST) Create(ctx genericapirequest.Context, obj runtime.Object, includeUninitialized bool) (runtime.Object, error) {
	return r.store.Create(ctx, obj, includeUninitialized)
}

func (r *REST) Update(ctx genericapirequest.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}

func (r *REST) Get(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

func (r *REST) Watch(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return r.store.Watch(ctx, options)
}

func (r *REST) Export(ctx genericapirequest.Context, name string, opts metav1.ExportOptions) (runtime.Object, error) {
	return r.store.Export(ctx, name, opts)
}

// Delete enforces life-cycle rules for namespace termination
func (r *REST) Delete(ctx genericapirequest.Context, name string, options *metav1.DeleteOptions) (runtime.Object, bool, error) {
	nsObj, err := r.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		return nil, false, err
	}

	namespace := nsObj.(*api.Namespace)

	// Ensure we have a UID precondition
	if options == nil {
		options = metav1.NewDeleteOptions(0)
	}
	if options.Preconditions == nil {
		options.Preconditions = &metav1.Preconditions{}
	}
	if options.Preconditions.UID == nil {
		options.Preconditions.UID = &namespace.UID
	} else if *options.Preconditions.UID != namespace.UID {
		err = apierrors.NewConflict(
			api.Resource("namespaces"),
			name,
			fmt.Errorf("Precondition failed: UID in precondition: %v, UID in object meta: %v", *options.Preconditions.UID, namespace.UID),
		)
		return nil, false, err
	}

	// upon first request to delete, we switch the phase to start namespace termination
	// TODO: enhance graceful deletion's calls to DeleteStrategy to allow phase change and finalizer patterns
	if namespace.DeletionTimestamp.IsZero() {
		key, err := r.store.KeyFunc(ctx, name)
		if err != nil {
			return nil, false, err
		}

		preconditions := storage.Preconditions{UID: options.Preconditions.UID}

		out := r.store.NewFunc()
		err = r.store.Storage.GuaranteedUpdate(
			ctx, key, out, false, &preconditions,
			storage.SimpleUpdate(func(existing runtime.Object) (runtime.Object, error) {
				existingNamespace, ok := existing.(*api.Namespace)
				if !ok {
					// wrong type
					return nil, fmt.Errorf("expected *api.Namespace, got %v", existing)
				}
				// Set the deletion timestamp if needed
				if existingNamespace.DeletionTimestamp.IsZero() {
					now := metav1.Now()
					existingNamespace.DeletionTimestamp = &now
				}
				// Set the namespace phase to terminating, if needed
				if existingNamespace.Status.Phase != api.NamespaceTerminating {
					existingNamespace.Status.Phase = api.NamespaceTerminating
				}

				// Remove orphan finalizer if options.OrphanDependents = false.
				if options.OrphanDependents != nil && *options.OrphanDependents == false {
					// remove Orphan finalizer.
					newFinalizers := []string{}
					for i := range existingNamespace.ObjectMeta.Finalizers {
						finalizer := existingNamespace.ObjectMeta.Finalizers[i]
						if string(finalizer) != metav1.FinalizerOrphanDependents {
							newFinalizers = append(newFinalizers, finalizer)
						}
					}
					existingNamespace.ObjectMeta.Finalizers = newFinalizers
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
			return nil, false, err
		}

		return out, false, nil
	}

	// prior to final deletion, we must ensure that finalizers is empty
	if len(namespace.Spec.Finalizers) != 0 {
		err = apierrors.NewConflict(api.Resource("namespaces"), namespace.Name, fmt.Errorf("The system is ensuring all content is removed from this namespace.  Upon completion, this namespace will automatically be purged by the system."))
		return nil, false, err
	}
	return r.store.Delete(ctx, name, options)
}

// Implement ShortNamesProvider
var _ rest.ShortNamesProvider = &REST{}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"ns"}
}

func (r *StatusREST) New() runtime.Object {
	return r.store.New()
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx genericapirequest.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}

func (r *FinalizeREST) New() runtime.Object {
	return r.store.New()
}

// Update alters the status finalizers subset of an object.
func (r *FinalizeREST) Update(ctx genericapirequest.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}
