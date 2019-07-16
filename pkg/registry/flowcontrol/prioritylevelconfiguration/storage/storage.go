/*
Copyright 2019 The Kubernetes Authors.

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
	"errors"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
	flowcontrolbootstrap "k8s.io/kubernetes/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/flowcontrol/flowschema"
	"k8s.io/kubernetes/pkg/registry/flowcontrol/prioritylevelconfiguration"
)

// Container includes dummy storage for RC pods and experimental storage for Scale.
type PriorityLevelConfigurationStorage struct {
	PriorityLevelConfiguration *REST
	Status                     *StatusREST
}

func NewStorage(optsGetter generic.RESTOptionsGetter) PriorityLevelConfigurationStorage {
	priorityLevelConfigurationREST, priorityLevelConfigurationStatusREST := NewREST(optsGetter)

	return PriorityLevelConfigurationStorage{
		PriorityLevelConfiguration: priorityLevelConfigurationREST,
		Status:                     priorityLevelConfigurationStatusREST,
	}
}

// REST implements a RESTStorage for jobs against etcd
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against Jobs.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, *StatusREST) {
	store := &genericregistry.Store{
		NewFunc:                  func() runtime.Object { return &flowcontrol.PriorityLevelConfiguration{} },
		NewListFunc:              func() runtime.Object { return &flowcontrol.PriorityLevelConfiguration{} },
		DefaultQualifiedResource: flowcontrol.Resource("prioritylevelconfigurations"),

		CreateStrategy: prioritylevelconfiguration.Strategy,
		UpdateStrategy: prioritylevelconfiguration.Strategy,
		DeleteStrategy: prioritylevelconfiguration.Strategy,

		// TODO: develop printer handler for prioritylevelconfiguration resource
		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter}
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}

	statusStore := *store
	statusStore.UpdateStrategy = flowschema.StatusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}
}

// Delete ensures that system priority classes are not deleted.
func (r *REST) Delete(ctx context.Context, name string, deleteValidation rest.ValidateObjectFunc, options *metav1.DeleteOptions) (runtime.Object, bool, error) {
	for _, pl := range flowcontrolbootstrap.SystemPriorityLevelConfigurations() {
		if name == pl.Name {
			return nil, false, apierrors.NewForbidden(flowcontrol.Resource("prioritylevelconfigurations"), pl.Name, errors.New("this is a system priority level and cannot be deleted"))
		}
	}

	return r.Store.Delete(ctx, name, deleteValidation, options)
}

// StatusREST implements the REST endpoint for changing the status of a resourcequota.
type StatusREST struct {
	store *genericregistry.Store
}

// New creates a new Job object.
func (r *StatusREST) New() runtime.Object {
	return &flowcontrol.FlowSchema{}
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
