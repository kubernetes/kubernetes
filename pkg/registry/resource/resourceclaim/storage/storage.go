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

package storage

import (
	"context"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/resource/resourceclaim"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
)

// REST implements a RESTStorage for ResourceClaims.
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against ResourceClaims.
func NewREST(optsGetter generic.RESTOptionsGetter, nsClient v1.NamespaceInterface) (*REST, *StatusREST, error) {
	if nsClient == nil {
		return nil, nil, fmt.Errorf("namespace client is required")
	}
	strategy := resourceclaim.NewStrategy(nsClient)
	store := &genericregistry.Store{
		NewFunc:                   func() runtime.Object { return &resource.ResourceClaim{} },
		NewListFunc:               func() runtime.Object { return &resource.ResourceClaimList{} },
		PredicateFunc:             resourceclaim.Match,
		DefaultQualifiedResource:  resource.Resource("resourceclaims"),
		SingularQualifiedResource: resource.Resource("resourceclaim"),

		CreateStrategy:      strategy,
		UpdateStrategy:      strategy,
		DeleteStrategy:      strategy,
		ReturnDeletedObject: true,
		ResetFieldsStrategy: strategy,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: resourceclaim.GetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, nil, err
	}

	statusStore := *store
	statusStrategy := resourceclaim.NewStatusStrategy(strategy)
	statusStore.UpdateStrategy = statusStrategy
	statusStore.ResetFieldsStrategy = statusStrategy

	rest := &REST{store}

	return rest, &StatusREST{store: &statusStore}, nil
}

// StatusREST implements the REST endpoint for changing the status of a ResourceClaim.
type StatusREST struct {
	store *genericregistry.Store
}

// New creates a new ResourceClaim object.
func (r *StatusREST) New() runtime.Object {
	return &resource.ResourceClaim{}
}

func (r *StatusREST) Destroy() {
	// Given that underlying store is shared with REST,
	// we don't destroy it here explicitly.
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	// We are explicitly setting forceAllowCreate to false in the call to the underlying storage because
	// subresources should never allow create on update.
	return r.store.Update(ctx, name, objInfo, createValidation, updateValidation, false, options)
}

// GetResetFields implements rest.ResetFieldsStrategy
func (r *StatusREST) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return r.store.GetResetFields()
}
