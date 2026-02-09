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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/admissionregistration/resolver"
	"k8s.io/kubernetes/pkg/registry/admissionregistration/validatingadmissionpolicy"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
)

// REST implements a RESTStorage for validatingAdmissionPolicy against etcd
type REST struct {
	*genericregistry.Store
}

// StatusREST implements a RESTStorage for ValidatingAdmissionPolicyStatus
type StatusREST struct {
	// DO NOT embed Store, manually select function to export.
	store *genericregistry.Store
}

var groupResource = admissionregistration.Resource("validatingadmissionpolicies")

// NewREST returns two RESTStorage objects that will work against validatingAdmissionPolicy and its status.
func NewREST(optsGetter generic.RESTOptionsGetter, authorizer authorizer.Authorizer, resourceResolver resolver.ResourceResolver) (*REST, *StatusREST, error) {
	r := &REST{}
	strategy := validatingadmissionpolicy.NewStrategy(authorizer, resourceResolver)
	store := &genericregistry.Store{
		NewFunc:     func() runtime.Object { return &admissionregistration.ValidatingAdmissionPolicy{} },
		NewListFunc: func() runtime.Object { return &admissionregistration.ValidatingAdmissionPolicyList{} },
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*admissionregistration.ValidatingAdmissionPolicy).Name, nil
		},
		DefaultQualifiedResource:  groupResource,
		SingularQualifiedResource: admissionregistration.Resource("validatingadmissionpolicy"),

		CreateStrategy:      strategy,
		UpdateStrategy:      strategy,
		DeleteStrategy:      strategy,
		ResetFieldsStrategy: strategy,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, nil, err
	}
	r.Store = store
	statusStrategy := validatingadmissionpolicy.NewStatusStrategy(strategy)
	statusStore := *store
	statusStore.UpdateStrategy = statusStrategy
	statusStore.ResetFieldsStrategy = statusStrategy
	sr := &StatusREST{store: &statusStore}
	return r, sr, nil
}

// Implement CategoriesProvider
var _ rest.CategoriesProvider = &REST{}

// Categories implements the CategoriesProvider interface. Returns a list of categories a resource is part of.
func (r *REST) Categories() []string {
	return []string{"api-extensions"}
}

// New generates a new ValidatingAdmissionPolicy object
func (r *StatusREST) New() runtime.Object {
	return &admissionregistration.ValidatingAdmissionPolicy{}
}

// Destroy disposes the store object. For the StatusREST, this is a no-op.
func (r *StatusREST) Destroy() {
	// Given that underlying store is shared with REST,
	// we don't destroy it here explicitly.
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

// GetResetFields returns the fields that got reset by the REST endpoint
func (r *StatusREST) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return r.store.GetResetFields()
}

// ConvertToTable delegates to the store, implements TableConverter
func (r *StatusREST) ConvertToTable(ctx context.Context, object runtime.Object, tableOptions runtime.Object) (*metav1.Table, error) {
	return r.store.ConvertToTable(ctx, object, tableOptions)
}

// Update alters the status subset of an object. Delegates to the store
func (r *StatusREST) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo, createValidation, updateValidation, forceAllowCreate, options)
}
