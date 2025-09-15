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
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/core/event"
)

// REST implements a RESTStorage for events.
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against events.
func NewREST(optsGetter generic.RESTOptionsGetter, ttl uint64) (*REST, error) {
	store := &genericregistry.Store{
		NewFunc:       func() runtime.Object { return &api.Event{} },
		NewListFunc:   func() runtime.Object { return &api.EventList{} },
		PredicateFunc: event.Matcher,
		TTLFunc: func(runtime.Object, uint64, bool) (uint64, error) {
			return ttl, nil
		},
		DefaultQualifiedResource:  api.Resource("events"),
		SingularQualifiedResource: api.Resource("event"),

		CreateStrategy: event.Strategy,
		UpdateStrategy: event.Strategy,
		DeleteStrategy: event.Strategy,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: event.GetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, err
	}
	return &REST{store}, nil
}

// Implement ShortNamesProvider
var _ rest.ShortNamesProvider = &REST{}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"ev"}
}
