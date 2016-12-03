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

package rest

import (
	"fmt"

	"k8s.io/kubernetes/cmd/libs/go2idl/client-gen/test_apis/testgroup"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	genericregistry "k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/storage/storagebackend"
)

type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work with testtype.
func NewREST(config *storagebackend.Config, storageDecorator generic.StorageDecorator) *REST {
	prefix := "/testtype"
	newListFunc := func() runtime.Object { return &testgroup.TestTypeList{} }
	// Usually you should reuse your RESTCreateStrategy.
	strategy := &NotNamespaceScoped{}
	getAttrs := func(obj runtime.Object) (labels.Set, fields.Set, error) {
		testObj, ok := obj.(*testgroup.TestType)
		if !ok {
			return nil, nil, fmt.Errorf("not a TestType")
		}
		return labels.Set(testObj.Labels), nil, nil
	}
	storageInterface, _ := storageDecorator(
		config, 100, &testgroup.TestType{}, prefix, strategy, newListFunc, getAttrs, storage.NoTriggerPublisher)
	store := &genericregistry.Store{
		NewFunc: func() runtime.Object { return &testgroup.TestType{} },
		// NewListFunc returns an object capable of storing results of an etcd list.
		NewListFunc: newListFunc,
		// Produces a path that etcd understands, to the root of the resource
		// by combining the namespace in the context with the given prefix.
		KeyRootFunc: func(ctx api.Context) string {
			return genericregistry.NamespaceKeyRootFunc(ctx, prefix)
		},
		// Produces a path that etcd understands, to the resource by combining
		// the namespace in the context with the given prefix.
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return genericregistry.NamespaceKeyFunc(ctx, prefix, name)
		},
		// Retrieve the name field of the resource.
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*testgroup.TestType).Name, nil
		},
		// Used to match objects based on labels/fields for list.
		PredicateFunc: func(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
			return storage.SelectionPredicate{
				Label: label,
				Field: field,
				GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
					testType, ok := obj.(*testgroup.TestType)
					if !ok {
						return nil, nil, fmt.Errorf("unexpected type of given object")
					}
					return labels.Set(testType.ObjectMeta.Labels), fields.Set{}, nil
				},
			}
		},
		Storage: storageInterface,
	}
	return &REST{store}
}

type NotNamespaceScoped struct {
}

func (*NotNamespaceScoped) NamespaceScoped() bool {
	return false
}
