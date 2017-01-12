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
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/registry/extensions/thirdpartyresourcedata"
	"k8s.io/kubernetes/pkg/registry/generic"
	genericregistry "k8s.io/kubernetes/pkg/registry/generic/registry"
)

// REST implements a RESTStorage for ThirdPartyResourceData
type REST struct {
	*genericregistry.Store
	kind string
}

// NewREST returns a registry which will store ThirdPartyResourceData in the given helper
func NewREST(optsGetter generic.RESTOptionsGetter, group, kind string) *REST {
	resource := extensions.Resource("thirdpartyresourcedatas")
	opts, err := optsGetter.GetRESTOptions(resource)
	if err != nil {
		panic(err) // TODO: Propagate error up
	}

	// We explicitly do NOT do any decoration here yet.
	opts.Decorator = generic.UndecoratedStorage // TODO use watchCacheSize=-1 to signal UndecoratedStorage
	opts.ResourcePrefix = "/ThirdPartyResourceData/" + group + "/" + strings.ToLower(kind) + "s"

	store := &genericregistry.Store{
		NewFunc:     func() runtime.Object { return &extensions.ThirdPartyResourceData{} },
		NewListFunc: func() runtime.Object { return &extensions.ThirdPartyResourceDataList{} },
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*extensions.ThirdPartyResourceData).Name, nil
		},
		PredicateFunc:     thirdpartyresourcedata.Matcher,
		QualifiedResource: resource,

		CreateStrategy: thirdpartyresourcedata.Strategy,
		UpdateStrategy: thirdpartyresourcedata.Strategy,
		DeleteStrategy: thirdpartyresourcedata.Strategy,
	}
	options := &generic.StoreOptions{RESTOptions: opts, AttrFunc: thirdpartyresourcedata.GetAttrs} // Pass in opts to use UndecoratedStorage and custom ResourcePrefix
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}

	return &REST{
		Store: store,
		kind:  kind,
	}
}

// Implements the rest.KindProvider interface
func (r *REST) Kind() string {
	return r.kind
}
