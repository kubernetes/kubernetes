/*
Copyright 2024 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/printers"
	"k8s.io/kubernetes/pkg/registry/storagemigration/storagemigration"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	svmapi "k8s.io/kubernetes/pkg/apis/storagemigration"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
)

// REST is a RESTStorage for ClusterTrustBundle.
type REST struct {
	*genericregistry.Store
}

type StatusREST struct {
	store *genericregistry.Store
}

var _ rest.StandardStorage = &REST{}
var _ rest.TableConvertor = &REST{}
var _ genericregistry.GenericStore = &REST{}

// NewREST returns a RESTStorage object for ClusterTrustBundle objects.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, *StatusREST, error) {
	store := &genericregistry.Store{
		NewFunc:     func() runtime.Object { return &svmapi.StorageVersionMigration{} },
		NewListFunc: func() runtime.Object { return &svmapi.StorageVersionMigrationList{} },
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*svmapi.StorageVersionMigration).Name, nil
		},
		DefaultQualifiedResource:  svmapi.Resource("storageversionmigrations"),
		SingularQualifiedResource: svmapi.Resource("storageversionmigration"),

		CreateStrategy:      storagemigration.Strategy,
		UpdateStrategy:      storagemigration.Strategy,
		DeleteStrategy:      storagemigration.Strategy,
		ResetFieldsStrategy: storagemigration.Strategy,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{
		RESTOptions: optsGetter,
	}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, nil, err
	}

	statusStore := *store
	statusStore.UpdateStrategy = storagemigration.StatusStrategy
	statusStore.ResetFieldsStrategy = storagemigration.StatusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}, nil
}

// New creates a new StorageVersion object.
func (r *StatusREST) New() runtime.Object {
	return &svmapi.StorageVersionMigration{}
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
