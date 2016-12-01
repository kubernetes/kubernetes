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
	"k8s.io/kubernetes/pkg/util/validation/field"
)

type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work with testtype.
func NewREST(config *storagebackend.Config, storageDecorator generic.StorageDecorator) *REST {
	opts := generic.RESTOptions{StorageConfig: config, Decorator: storageDecorator, ResourcePrefix: "/testtype", DeleteCollectionWorkers: 1}
	store := &genericregistry.Store{
		NewFunc: func() runtime.Object { return &testgroup.TestType{} },
		// NewListFunc returns an object capable of storing results of an etcd list.
		NewListFunc: func() runtime.Object { return &testgroup.TestTypeList{} },
		// Produces a path that etcd understands, to the root of the resource
		// by combining the namespace in the context with the given prefix.
		// This func can be omitted when the name of the object is enough to locate it
		KeyRootFunc: func(ctx api.Context) string {
			return opts.ResourcePrefix
		},
		// Produces a path that etcd understands, to the resource by combining
		// the namespace in the context with the given prefix.
		// This func can be omitted when the name of the object is enough to locate it
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return genericregistry.NoNamespaceKeyFunc(ctx, opts.ResourcePrefix, name)
		},
		// Retrieve the name field of the resource.
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*testgroup.TestType).Name, nil
		},
		QualifiedResource: api.Resource("testtype"),
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
		CreateStrategy: &fakeStrategy{api.Scheme, api.SimpleNameGenerator},
	}
	getAttrs := func(obj runtime.Object) (labels.Set, fields.Set, error) {
		testObj, ok := obj.(*testgroup.TestType)
		if !ok {
			return nil, nil, fmt.Errorf("not a TestType")
		}
		return labels.Set(testObj.Labels), nil, nil
	}
	options := &generic.StoreOptions{RESTOptions: opts, AttrFunc: getAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}
	return &REST{store}
}

type fakeStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

func (*fakeStrategy) NamespaceScoped() bool                                        { return false }
func (*fakeStrategy) PrepareForCreate(ctx api.Context, obj runtime.Object)         {}
func (*fakeStrategy) Validate(ctx api.Context, obj runtime.Object) field.ErrorList { return nil }
func (*fakeStrategy) Canonicalize(obj runtime.Object)                              {}
