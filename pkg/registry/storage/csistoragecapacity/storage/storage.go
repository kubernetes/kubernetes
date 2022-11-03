/*
Copyright 2020 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	storageapi "k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/registry/storage/csistoragecapacity"
)

// CSIStorageCapacityStorage includes storage for CSIStorageCapacity and all subresources
type CSIStorageCapacityStorage struct {
	CSIStorageCapacity *REST
}

// REST object that will work for CSIStorageCapacity
type REST struct {
	*genericregistry.Store
}

// NewStorage returns a RESTStorage object that will work against CSIStorageCapacity
func NewStorage(optsGetter generic.RESTOptionsGetter) (*CSIStorageCapacityStorage, error) {
	store := &genericregistry.Store{
		NewFunc:                   func() runtime.Object { return &storageapi.CSIStorageCapacity{} },
		NewListFunc:               func() runtime.Object { return &storageapi.CSIStorageCapacityList{} },
		DefaultQualifiedResource:  storageapi.Resource("csistoragecapacities"),
		SingularQualifiedResource: storageapi.Resource("csistoragecapacity"),

		TableConvertor: rest.NewDefaultTableConvertor(storageapi.Resource("csistoragecapacities")),

		CreateStrategy: csistoragecapacity.Strategy,
		UpdateStrategy: csistoragecapacity.Strategy,
		DeleteStrategy: csistoragecapacity.Strategy,
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, err
	}

	return &CSIStorageCapacityStorage{
		CSIStorageCapacity: &REST{store},
	}, nil
}
