/*
Copyright 2017 The Kubernetes Authors.

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

package customresourcedefinition

import (
	"fmt"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	storageerr "k8s.io/apiserver/pkg/storage/errors"
)

// rest implements a RESTStorage for API services against etcd
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against API services.
func NewREST(scheme *runtime.Scheme, optsGetter generic.RESTOptionsGetter) *REST {
	strategy := NewStrategy(scheme)

	store := &genericregistry.Store{
		Copier:                   scheme,
		NewFunc:                  func() runtime.Object { return &apiextensions.CustomResourceDefinition{} },
		NewListFunc:              func() runtime.Object { return &apiextensions.CustomResourceDefinitionList{} },
		PredicateFunc:            MatchCustomResourceDefinition,
		DefaultQualifiedResource: apiextensions.Resource("customresourcedefinitions"),

		CreateStrategy: strategy,
		UpdateStrategy: strategy,
		DeleteStrategy: strategy,
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: GetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}
	return &REST{store}
}

// Implement ShortNamesProvider
var _ rest.ShortNamesProvider = &REST{}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"crd"}
}

// Delete adds the CRD finalizer to the list
func (r *REST) Delete(ctx genericapirequest.Context, name string, options *metav1.DeleteOptions) (runtime.Object, bool, error) {
	obj, err := r.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		return nil, false, err
	}

	crd := obj.(*apiextensions.CustomResourceDefinition)

	// Ensure we have a UID precondition
	if options == nil {
		options = metav1.NewDeleteOptions(0)
	}
	if options.Preconditions == nil {
		options.Preconditions = &metav1.Preconditions{}
	}
	if options.Preconditions.UID == nil {
		options.Preconditions.UID = &crd.UID
	} else if *options.Preconditions.UID != crd.UID {
		err = apierrors.NewConflict(
			apiextensions.Resource("customresourcedefinitions"),
			name,
			fmt.Errorf("Precondition failed: UID in precondition: %v, UID in object meta: %v", *options.Preconditions.UID, crd.UID),
		)
		return nil, false, err
	}

	// upon first request to delete, add our finalizer and then delegate
	if crd.DeletionTimestamp.IsZero() {
		key, err := r.Store.KeyFunc(ctx, name)
		if err != nil {
			return nil, false, err
		}

		preconditions := storage.Preconditions{UID: options.Preconditions.UID}

		out := r.Store.NewFunc()
		err = r.Store.Storage.GuaranteedUpdate(
			ctx, key, out, false, &preconditions,
			storage.SimpleUpdate(func(existing runtime.Object) (runtime.Object, error) {
				existingCRD, ok := existing.(*apiextensions.CustomResourceDefinition)
				if !ok {
					// wrong type
					return nil, fmt.Errorf("expected *apiextensions.CustomResourceDefinition, got %v", existing)
				}

				// Set the deletion timestamp if needed
				if existingCRD.DeletionTimestamp.IsZero() {
					now := metav1.Now()
					existingCRD.DeletionTimestamp = &now
				}

				if !apiextensions.CRDHasFinalizer(existingCRD, apiextensions.CustomResourceCleanupFinalizer) {
					existingCRD.Finalizers = append(existingCRD.Finalizers, apiextensions.CustomResourceCleanupFinalizer)
				}
				// update the status condition too
				apiextensions.SetCRDCondition(existingCRD, apiextensions.CustomResourceDefinitionCondition{
					Type:    apiextensions.Terminating,
					Status:  apiextensions.ConditionTrue,
					Reason:  "InstanceDeletionPending",
					Message: "CustomResourceDefinition marked for deletion; CustomResource deletion will begin soon",
				})
				return existingCRD, nil
			}),
		)

		if err != nil {
			err = storageerr.InterpretGetError(err, apiextensions.Resource("customresourcedefinitions"), name)
			err = storageerr.InterpretUpdateError(err, apiextensions.Resource("customresourcedefinitions"), name)
			if _, ok := err.(*apierrors.StatusError); !ok {
				err = apierrors.NewInternalError(err)
			}
			return nil, false, err
		}

		return out, false, nil
	}

	return r.Store.Delete(ctx, name, options)
}

// NewStatusREST makes a RESTStorage for status that has more limited options.
// It is based on the original REST so that we can share the same underlying store
func NewStatusREST(scheme *runtime.Scheme, rest *REST) *StatusREST {
	statusStore := *rest.Store
	statusStore.CreateStrategy = nil
	statusStore.DeleteStrategy = nil
	statusStore.UpdateStrategy = NewStatusStrategy(scheme)
	return &StatusREST{store: &statusStore}
}

type StatusREST struct {
	store *genericregistry.Store
}

var _ = rest.Updater(&StatusREST{})

func (r *StatusREST) New() runtime.Object {
	return &apiextensions.CustomResourceDefinition{}
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx genericapirequest.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}
