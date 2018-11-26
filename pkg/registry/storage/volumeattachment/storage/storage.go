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

package storage

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	storageapi "k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/registry/storage/volumeattachment"
)

// VolumeAttachmentStorage includes storage for VolumeAttachments and all subresources
type VolumeAttachmentStorage struct {
	VolumeAttachment *REST
	Status           *StatusREST
}

// REST object that will work for VolumeAttachments
type REST struct {
	*genericregistry.Store
}

// NewStorage returns a RESTStorage object that will work against VolumeAttachments
func NewStorage(optsGetter generic.RESTOptionsGetter) *VolumeAttachmentStorage {
	store := &genericregistry.Store{
		NewFunc:                  func() runtime.Object { return &storageapi.VolumeAttachment{} },
		NewListFunc:              func() runtime.Object { return &storageapi.VolumeAttachmentList{} },
		DefaultQualifiedResource: storageapi.Resource("volumeattachments"),

		CreateStrategy:      volumeattachment.Strategy,
		UpdateStrategy:      volumeattachment.Strategy,
		DeleteStrategy:      volumeattachment.Strategy,
		ReturnDeletedObject: true,
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter}
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}

	statusStore := *store
	statusStore.UpdateStrategy = volumeattachment.StatusStrategy

	return &VolumeAttachmentStorage{
		VolumeAttachment: &REST{store},
		Status:           &StatusREST{store: &statusStore},
	}
}

// StatusREST implements the REST endpoint for changing the status of a VolumeAttachment
type StatusREST struct {
	store *genericregistry.Store
}

// New creates a new VolumeAttachment resource
func (r *StatusREST) New() runtime.Object {
	return &storageapi.VolumeAttachment{}
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
