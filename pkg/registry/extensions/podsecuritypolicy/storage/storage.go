/*
Copyright 2014 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/registry/extensions/podsecuritypolicy"
)

// REST implements a RESTStorage for PodSecurityPolicies.
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against PodSecurityPolicy objects.
func NewREST(optsGetter generic.RESTOptionsGetter) *REST {
	store := &genericregistry.Store{
		Copier:                   api.Scheme,
		NewFunc:                  func() runtime.Object { return &extensions.PodSecurityPolicy{} },
		NewListFunc:              func() runtime.Object { return &extensions.PodSecurityPolicyList{} },
		DefaultQualifiedResource: extensions.Resource("podsecuritypolicies"),
		WatchCacheSize:           cachesize.GetWatchCacheSizeByResource("podsecuritypolicies"),

		CreateStrategy:      podsecuritypolicy.Strategy,
		UpdateStrategy:      podsecuritypolicy.Strategy,
		DeleteStrategy:      podsecuritypolicy.Strategy,
		ReturnDeletedObject: true,
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter}
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}
	return &REST{store}
}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"psp"}
}
