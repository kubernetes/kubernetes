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

package wardle

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/sample-apiserver/pkg/apis/wardle"
)

// rest implements a RESTStorage for API services against etcd
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against API services.
func NewREST(scheme *runtime.Scheme, optsGetter generic.RESTOptionsGetter) *REST {
	strategy := NewStrategy(scheme)

	store := &genericregistry.Store{
		Copier:            scheme,
		NewFunc:           func() runtime.Object { return &wardle.Flunder{} },
		NewListFunc:       func() runtime.Object { return &wardle.FlunderList{} },
		PredicateFunc:     MatchFlunder,
		QualifiedResource: wardle.Resource("flunders"),

		CreateStrategy: strategy,
		UpdateStrategy: strategy,
		DeleteStrategy: strategy,
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: GetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}
	return &REST{store}
}
