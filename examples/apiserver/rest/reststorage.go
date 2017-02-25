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

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/cmd/libs/go2idl/client-gen/test_apis/testgroup"
	"k8s.io/kubernetes/pkg/api"
)

type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work with testtype.
func NewREST(optsGetter generic.RESTOptionsGetter) *REST {
	store := &genericregistry.Store{
		Copier:  api.Scheme,
		NewFunc: func() runtime.Object { return &testgroup.TestType{} },
		// NewListFunc returns an object capable of storing results of an etcd list.
		NewListFunc: func() runtime.Object { return &testgroup.TestTypeList{} },
		// Retrieve the name field of the resource.
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*testgroup.TestType).Name, nil
		},
		// Used to match objects based on labels/fields for list.
		PredicateFunc: matcher,
		// QualifiedResource should always be plural
		QualifiedResource: api.Resource("testtypes"),

		CreateStrategy: strategy,
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: getAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}
	return &REST{store}
}

type fakeStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

func (*fakeStrategy) NamespaceScoped() bool                                              { return false }
func (*fakeStrategy) PrepareForCreate(ctx genericapirequest.Context, obj runtime.Object) {}
func (*fakeStrategy) Validate(ctx genericapirequest.Context, obj runtime.Object) field.ErrorList {
	return nil
}
func (*fakeStrategy) Canonicalize(obj runtime.Object) {}

var strategy = &fakeStrategy{api.Scheme, names.SimpleNameGenerator}

func getAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	testType, ok := obj.(*testgroup.TestType)
	if !ok {
		return nil, nil, fmt.Errorf("not a TestType")
	}
	return labels.Set(testType.ObjectMeta.Labels), fields.Set{}, nil
}

func matcher(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: getAttrs,
	}
}
