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

package etcd

import (
	"k8s.io/kubernetes/pkg/api"
	extensionsapi "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/extensions/networkpolicy"
	"k8s.io/kubernetes/pkg/registry/generic"
	genericregistry "k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// rest implements a RESTStorage for network policies against etcd
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against network policies.
func NewREST(opts generic.RESTOptions) *REST {
	prefix := "/" + opts.ResourcePrefix

	newListFunc := func() runtime.Object { return &extensionsapi.NetworkPolicyList{} }
	storageInterface, dFunc := opts.Decorator(
		opts.StorageConfig,
		cachesize.GetWatchCacheSizeByResource(cachesize.NetworkPolicys),
		&extensionsapi.NetworkPolicy{},
		prefix,
		networkpolicy.Strategy,
		newListFunc,
		networkpolicy.GetAttrs,
		storage.NoTriggerPublisher,
	)

	store := &genericregistry.Store{
		NewFunc: func() runtime.Object { return &extensionsapi.NetworkPolicy{} },

		// NewListFunc returns an object capable of storing results of an etcd list.
		NewListFunc: newListFunc,
		// Produces a NetworkPolicy that etcd understands, to the root of the resource
		// by combining the namespace in the context with the given prefix
		KeyRootFunc: func(ctx api.Context) string {
			return genericregistry.NamespaceKeyRootFunc(ctx, prefix)
		},
		// Produces a NetworkPolicy that etcd understands, to the resource by combining
		// the namespace in the context with the given prefix
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return genericregistry.NamespaceKeyFunc(ctx, prefix, name)
		},
		// Retrieve the name field of a network policy
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*extensionsapi.NetworkPolicy).Name, nil
		},
		// Used to match objects based on labels/fields for list and watch
		PredicateFunc:           networkpolicy.MatchNetworkPolicy,
		QualifiedResource:       extensionsapi.Resource("networkpolicies"),
		EnableGarbageCollection: opts.EnableGarbageCollection,
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,

		// Used to validate controller creation
		CreateStrategy: networkpolicy.Strategy,

		// Used to validate controller updates
		UpdateStrategy: networkpolicy.Strategy,
		DeleteStrategy: networkpolicy.Strategy,

		Storage:     storageInterface,
		DestroyFunc: dFunc,
	}
	return &REST{store}
}
