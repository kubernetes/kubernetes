/*
Copyright 2016 The Kubernetes Authors.

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

package apiservice

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"

	"k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/apis/apiregistration"
)

// rest implements a RESTStorage for network policies against etcd
type REST struct {
	*registry.Store
}

// NewREST returns a RESTStorage object that will work against network policies.
func NewREST(opts generic.RESTOptions) *REST {
	prefix := "/" + opts.ResourcePrefix

	newListFunc := func() runtime.Object { return &apiregistration.APIServiceList{} }
	storageInterface, dFunc := opts.Decorator(
		opts.StorageConfig,
		1000, // cache size
		&apiregistration.APIService{},
		prefix,
		strategy,
		newListFunc,
		getAttrs,
		storage.NoTriggerPublisher,
	)

	store := &registry.Store{
		NewFunc: func() runtime.Object { return &apiregistration.APIService{} },

		// NewListFunc returns an object capable of storing results of an etcd list.
		NewListFunc: newListFunc,
		// Produces a APIService that etcd understands, to the root of the resource
		// by combining the namespace in the context with the given prefix
		KeyRootFunc: func(ctx api.Context) string {
			return prefix
		},
		// Produces a APIService that etcd understands, to the resource by combining
		// the namespace in the context with the given prefix
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return registry.NoNamespaceKeyFunc(ctx, prefix, name)
		},
		// Retrieve the name field of an apiserver
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*apiregistration.APIService).Name, nil
		},
		// Used to match objects based on labels/fields for list and watch
		PredicateFunc:           MatchAPIService,
		QualifiedResource:       apiregistration.Resource("apiservers"),
		EnableGarbageCollection: opts.EnableGarbageCollection,
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,

		// Used to validate controller creation
		CreateStrategy: strategy,

		// Used to validate controller updates
		UpdateStrategy: strategy,
		DeleteStrategy: strategy,

		Storage:     storageInterface,
		DestroyFunc: dFunc,
	}
	return &REST{store}
}

// getAttrs returns labels and fields of a given object for filtering purposes.
func getAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	castObj, ok := obj.(*apiregistration.APIService)
	if !ok {
		return nil, nil, fmt.Errorf("given object is not an APIService.")
	}
	return labels.Set(castObj.ObjectMeta.Labels), APIServiceToSelectableFields(castObj), nil
}
