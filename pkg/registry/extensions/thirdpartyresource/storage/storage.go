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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/extensions/thirdpartyresource"
)

// REST implements a RESTStorage for ThirdPartyResources
type REST struct {
	*genericregistry.Store
}

// NewREST returns a registry which will store ThirdPartyResource in the given helper
func NewREST(optsGetter generic.RESTOptionsGetter) *REST {
	resource := extensions.Resource("thirdpartyresources")
	opts, err := optsGetter.GetRESTOptions(resource)
	if err != nil {
		panic(err) // TODO: Propagate error up
	}

	// We explicitly do NOT do any decoration here yet. // TODO determine why we do not want to cache here
	opts.Decorator = generic.UndecoratedStorage

	store := &genericregistry.Store{
		Copier:            api.Scheme,
		NewFunc:           func() runtime.Object { return &extensions.ThirdPartyResource{} },
		NewListFunc:       func() runtime.Object { return &extensions.ThirdPartyResourceList{} },
		PredicateFunc:     thirdpartyresource.Matcher,
		QualifiedResource: resource,
		WatchCacheSize:    cachesize.GetWatchCacheSizeByResource(resource.Resource),

		CreateStrategy: thirdpartyresource.Strategy,
		UpdateStrategy: thirdpartyresource.Strategy,
		DeleteStrategy: thirdpartyresource.Strategy,
	}
	options := &generic.StoreOptions{RESTOptions: opts, AttrFunc: thirdpartyresource.GetAttrs} // Pass in opts to use UndecoratedStorage
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}
	return &REST{store}
}
