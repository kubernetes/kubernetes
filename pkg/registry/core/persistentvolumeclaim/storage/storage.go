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
	"context"

	pvcutil "k8s.io/kubernetes/pkg/api/persistentvolumeclaim"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/core/persistentvolumeclaim"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// REST implements a RESTStorage for persistent volume claims.
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against persistent volume claims.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, *StatusREST, error) {
	store := &genericregistry.Store{
		NewFunc:                   func() runtime.Object { return &api.PersistentVolumeClaim{} },
		NewListFunc:               func() runtime.Object { return &api.PersistentVolumeClaimList{} },
		PredicateFunc:             storage.PredicateFuncFromMatcherFunc(api.PersistentVolumeClaimMatcher),
		DefaultQualifiedResource:  api.Resource("persistentvolumeclaims"),
		SingularQualifiedResource: api.Resource("persistentvolumeclaim"),

		CreateStrategy:      persistentvolumeclaim.Strategy,
		UpdateStrategy:      persistentvolumeclaim.Strategy,
		DeleteStrategy:      persistentvolumeclaim.Strategy,
		ReturnDeletedObject: true,
		ResetFieldsStrategy: persistentvolumeclaim.Strategy,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: api.PersistentVolumeClaimGetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, nil, err
	}

	statusStore := *store
	statusStore.UpdateStrategy = persistentvolumeclaim.StatusStrategy
	statusStore.ResetFieldsStrategy = persistentvolumeclaim.StatusStrategy

	rest := &REST{store}
	store.Decorator = rest.defaultOnRead

	return rest, &StatusREST{store: &statusStore}, nil
}

// Implement ShortNamesProvider
var _ rest.ShortNamesProvider = &REST{}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"pvc"}
}

// defaultOnRead sets interlinked fields that were not previously set on read.
// We can't do this in the normal defaulting path because that same logic
// applies on Get, Create, and Update, but we need to distinguish between them.
//
// This will be called on both PersistentVolumeClaim and PersistentVolumeClaimList types.
func (r *REST) defaultOnRead(obj runtime.Object) {
	switch s := obj.(type) {
	case *api.PersistentVolumeClaim:
		r.defaultOnReadPvc(s)
	case *api.PersistentVolumeClaimList:
		r.defaultOnReadPvcList(s)
	default:
		// This was not an object we can default.  This is not an error, as the
		// caching layer can pass through here, too.
	}
}

// defaultOnReadPvcList defaults a PersistentVolumeClaimList.
func (r *REST) defaultOnReadPvcList(pvcList *api.PersistentVolumeClaimList) {
	if pvcList == nil {
		return
	}

	for i := range pvcList.Items {
		r.defaultOnReadPvc(&pvcList.Items[i])
	}
}

// defaultOnReadPvc defaults a single PersistentVolumeClaim.
func (r *REST) defaultOnReadPvc(pvc *api.PersistentVolumeClaim) {
	if pvc == nil {
		return
	}

	// We set dataSourceRef to the same value as dataSource at creation time now,
	// but for pre-existing PVCs with data sources, the dataSourceRef field will
	// be blank, so we fill it in here at read time.
	pvcutil.NormalizeDataSources(&pvc.Spec)
}

// StatusREST implements the REST endpoint for changing the status of a persistentvolumeclaim.
type StatusREST struct {
	store *genericregistry.Store
}

// New creates a new PersistentVolumeClaim object.
func (r *StatusREST) New() runtime.Object {
	return &api.PersistentVolumeClaim{}
}

// Destroy cleans up resources on shutdown.
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

func (r *StatusREST) ConvertToTable(ctx context.Context, object runtime.Object, tableOptions runtime.Object) (*metav1.Table, error) {
	return r.store.ConvertToTable(ctx, object, tableOptions)
}
