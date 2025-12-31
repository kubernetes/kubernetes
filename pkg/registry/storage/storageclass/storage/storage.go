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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	storageapi "k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/registry/storage/storageclass"
)

// REST implements a RESTStorage for storage classes.
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against storage classes.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, error) {
	tableConvertor := storageclass.NewTableConvertor()

	store := &genericregistry.Store{
		NewFunc:                   func() runtime.Object { return &storageapi.StorageClass{} },
		NewListFunc:               func() runtime.Object { return &storageapi.StorageClassList{} },
		DefaultQualifiedResource:  storageapi.Resource("storageclasses"),
		SingularQualifiedResource: storageapi.Resource("storageclass"),

		CreateStrategy:      storageclass.Strategy,
		UpdateStrategy:      storageclass.Strategy,
		DeleteStrategy:      storageclass.Strategy,
		ReturnDeletedObject: true,

		TableConvertor: tableConvertor,
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, err
	}

	r := &REST{Store: store}
	// Set lister after REST is created to break circular dependency
	tableConvertor.SetLister(r)
	return r, nil
}

// Implement ShortNamesProvider
var _ rest.ShortNamesProvider = &REST{}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"sc"}
}
