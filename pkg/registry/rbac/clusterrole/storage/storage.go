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

package storage

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/rbac/clusterrole"
)

// REST implements a RESTStorage for ClusterRole
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against ClusterRole objects.
func NewREST(optsGetter generic.RESTOptionsGetter) *REST {
	store := &genericregistry.Store{
		Copier:            api.Scheme,
		NewFunc:           func() runtime.Object { return &rbac.ClusterRole{} },
		NewListFunc:       func() runtime.Object { return &rbac.ClusterRoleList{} },
		PredicateFunc:     clusterrole.Matcher,
		QualifiedResource: rbac.Resource("clusterroles"),
		WatchCacheSize:    cachesize.GetWatchCacheSizeByResource("clusterroles"),

		CreateStrategy: clusterrole.Strategy,
		UpdateStrategy: clusterrole.Strategy,
		DeleteStrategy: clusterrole.Strategy,
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: clusterrole.GetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}

	return &REST{store}
}
