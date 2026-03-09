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
	"k8s.io/kubernetes/pkg/registry/checkpoint/podrestore"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
)

// REST implements a RESTStorage for PodRestores against etcd.
type REST struct {
	*genericregistry.Store
}

// StatusREST implements the REST endpoint for changing the status of a PodRestore.
type StatusREST struct {
	store *genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against PodRestores.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, *StatusREST, error) {
	store := &genericregistry.Store{
		NewFunc:                   func() runtime.Object { return &checkpointapi.PodRestore{} },
		NewListFunc:               func() runtime.Object { return &checkpointapi.PodRestoreList{} },
		DefaultQualifiedResource:  checkpointapi.Resource("podrestores"),
		SingularQualifiedResource: checkpointapi.Resource("podrestore"),

		CreateStrategy:      podrestore.Strategy,
		UpdateStrategy:      podrestore.Strategy,
		DeleteStrategy:      podrestore.Strategy,
		ResetFieldsStrategy: podrestore.Strategy,

		TableConvertor: rest.NewDefaultTableConvertor(checkpointapi.Resource("podrestores")),
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, nil, err
	}

	statusStore := *store
	statusStore.UpdateStrategy = podrestore.StatusStrategy
	statusStore.ResetFieldsStrategy = podrestore.StatusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}, nil
}

func (r *StatusREST) New() runtime.Object {
	return &checkpointapi.PodRestore{}
}

func (r *StatusREST) Destroy() {
}

func (r *StatusREST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

func (r *StatusREST) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo, createValidation, updateValidation, forceAllowCreate, options)
}

func (r *StatusREST) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return r.store.GetResetFields()
}
