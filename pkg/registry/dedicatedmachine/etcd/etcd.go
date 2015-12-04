/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package etcd

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/dedicatedmachine"
	"k8s.io/kubernetes/pkg/registry/generic"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// rest implements a RESTStorage for DedicatedMachines against etcd
type REST struct {
	*etcdgeneric.Etcd
}

// NewREST returns a RESTStorage object that will work against DedicatedMachines.
func NewREST(s storage.Interface, storageDecorator generic.StorageDecorator) *REST {
	prefix := "/dedicatedmachines"

	newListFunc := func() runtime.Object { return &extensions.DedicatedMachineList{} }
	storageInterface := storageDecorator(
		s, 100, &extensions.DedicatedMachine{}, prefix, false, newListFunc)

	store := &etcdgeneric.Etcd{
		NewFunc: func() runtime.Object { return &extensions.DedicatedMachine{} },

		// NewListFunc returns an object capable of storing results of an etcd list.
		NewListFunc: newListFunc,
		// Produces a path that etcd understands, to the root of the resource
		// by combining the namespace in the context with the given prefix
		KeyRootFunc: func(ctx api.Context) string {
			return etcdgeneric.NamespaceKeyRootFunc(ctx, prefix)
		},
		// Produces a path that etcd understands, to the resource by combining
		// the namespace in the context with the given prefix
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return etcdgeneric.NamespaceKeyFunc(ctx, prefix, name)
		},
		// Retrieve the name field of a dedicated machine
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*extensions.DedicatedMachine).Name, nil
		},
		// Used to match objects based on labels/fields for list and watch
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return dedicatedmachine.MatchDedicatedMachine(label, field)
		},
		EndpointName: "dedicatedmachines",

		// Used to validate dedicated machine creation
		CreateStrategy: dedicatedmachine.Strategy,

		// Used to validate dedicated machine updates
		UpdateStrategy: dedicatedmachine.Strategy,

		Storage: storageInterface,
	}
	return &REST{store}
}
