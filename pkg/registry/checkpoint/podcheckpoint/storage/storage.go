/*
Copyright 2026 The Kubernetes Authors.

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
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	checkpointapi "k8s.io/kubernetes/pkg/apis/checkpoint"
	"k8s.io/kubernetes/pkg/registry/checkpoint/podcheckpoint"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
)

// REST implements a RESTStorage for PodCheckpoints against etcd.
type REST struct {
	*genericregistry.Store
}

// StatusREST implements the REST endpoint for changing the status of a PodCheckpoint.
type StatusREST struct {
	store *genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against PodCheckpoints.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, *StatusREST, error) {
	store := &genericregistry.Store{
		NewFunc:                   func() runtime.Object { return &checkpointapi.PodCheckpoint{} },
		NewListFunc:               func() runtime.Object { return &checkpointapi.PodCheckpointList{} },
		DefaultQualifiedResource:  checkpointapi.Resource("podcheckpoints"),
		SingularQualifiedResource: checkpointapi.Resource("podcheckpoint"),

		CreateStrategy:      podcheckpoint.Strategy,
		UpdateStrategy:      podcheckpoint.Strategy,
		DeleteStrategy:      podcheckpoint.Strategy,
		ResetFieldsStrategy: podcheckpoint.Strategy,

		TableConvertor: rest.NewDefaultTableConvertor(checkpointapi.Resource("podcheckpoints")),
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, nil, err
	}

	statusStore := *store
	statusStore.UpdateStrategy = podcheckpoint.StatusStrategy
	statusStore.ResetFieldsStrategy = podcheckpoint.StatusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}, nil
}

// New creates a new PodCheckpoint object.
func (r *StatusREST) New() runtime.Object {
	return &checkpointapi.PodCheckpoint{}
}

// Destroy cleans up resources on shutdown.
func (r *StatusREST) Destroy() {
	// Given that underlying store is shared with REST, we don't need to destroy it here.
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo, createValidation, updateValidation, forceAllowCreate, options)
}

// GetResetFields implements rest.ResetFieldsStrategy
func (r *StatusREST) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return r.store.GetResetFields()
}
