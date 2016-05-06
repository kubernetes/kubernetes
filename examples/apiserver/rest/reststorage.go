/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/cmd/libs/go2idl/client-gen/test_apis/testgroup.k8s.io"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

type REST struct {
	*registry.Store
}

// NewREST returns a RESTStorage object that will work with testtype.
func NewREST(s storage.Interface, storageDecorator generic.StorageDecorator) *REST {
	prefix := "/testtype"
	newListFunc := func() runtime.Object { return &testgroup.TestTypeList{} }
	// Usually you should reuse your RESTCreateStrategy.
	strategy := &NotNamespaceScoped{}
	storageInterface := storageDecorator(
		s, 100, &testgroup.TestType{}, prefix, strategy, newListFunc)
	store := &registry.Store{
		NewFunc: func() runtime.Object { return &testgroup.TestType{} },
		// NewListFunc returns an object capable of storing results of an etcd list.
		NewListFunc: newListFunc,
		// Produces a path that etcd understands, to the root of the resource
		// by combining the namespace in the context with the given prefix.
		KeyRootFunc: func(ctx api.Context) string {
			return registry.NamespaceKeyRootFunc(ctx, prefix)
		},
		// Produces a path that etcd understands, to the resource by combining
		// the namespace in the context with the given prefix.
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return registry.NamespaceKeyFunc(ctx, prefix, name)
		},
		// Retrieve the name field of the resource.
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*testgroup.TestType).Name, nil
		},
		// Used to match objects based on labels/fields for list.
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return generic.MatcherFunc(nil)
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
