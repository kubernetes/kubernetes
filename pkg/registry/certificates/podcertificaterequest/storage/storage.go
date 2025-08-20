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
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	api "k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/certificates/podcertificaterequest"
	"k8s.io/utils/clock"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
)

// REST is a RESTStorage for PodCertificateRequest.
type REST struct {
	*genericregistry.Store
}

var _ rest.StandardStorage = &REST{}
var _ rest.TableConvertor = &REST{}
var _ genericregistry.GenericStore = &REST{}

// NewREST returns a RESTStorage object for PodCertificateRequest objects.
func NewREST(optsGetter generic.RESTOptionsGetter, authorizer authorizer.Authorizer, clock clock.PassiveClock) (*REST, *StatusREST, error) {
	strategy := podcertificaterequest.NewStrategy()

	store := &genericregistry.Store{
		NewFunc:                   func() runtime.Object { return &api.PodCertificateRequest{} },
		NewListFunc:               func() runtime.Object { return &api.PodCertificateRequestList{} },
		DefaultQualifiedResource:  api.Resource("podcertificaterequests"),
		SingularQualifiedResource: api.Resource("podcertificaterequest"),

		CreateStrategy: strategy,
		UpdateStrategy: strategy,
		DeleteStrategy: strategy,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{
		RESTOptions: optsGetter,
		AttrFunc:    getAttrs,
	}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, nil, err
	}

	statusStrategy := podcertificaterequest.NewStatusStrategy(strategy, authorizer, clock)

	// Subresources use the same store and creation strategy, which only
	// allows empty subs. Updates to an existing subresource are handled by
	// dedicated strategies.
	statusStore := *store
	statusStore.UpdateStrategy = statusStrategy
	statusStore.ResetFieldsStrategy = statusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}, nil
}

func getAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	req, ok := obj.(*api.PodCertificateRequest)
	if !ok {
		return nil, nil, fmt.Errorf("not a podcertificaterequest")
	}

	selectableFields := generic.MergeFieldsSets(generic.ObjectMetaFieldsSet(&req.ObjectMeta, true), fields.Set{
		"spec.signerName": req.Spec.SignerName,
		"spec.podName":    req.Spec.PodName,
		"spec.nodeName":   string(req.Spec.NodeName),
	})

	return labels.Set(req.Labels), selectableFields, nil
}

// StatusREST implements the REST endpoint for changing the status of a PodCertificateRequest.
type StatusREST struct {
	store *genericregistry.Store
}

// New creates a new PodCertificateRequest object.
func (r *StatusREST) New() runtime.Object {
	return &api.PodCertificateRequest{}
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

var _ = rest.Patcher(&StatusREST{})
