/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/registry/networkpolicy"
	"k8s.io/kubernetes/pkg/runtime"
)

// rest implements a RESTStorage for network policies against etcd
type REST struct {
	*registry.Store
}

// NewREST returns a RESTStorage object that will work against network policies.
func NewREST(opts generic.RESTOptions) *REST {
	prefix := "/networkpolicies"

	newListFunc := func() runtime.Object { return &extensionsapi.NetworkPolicyList{} }
	storageInterface := opts.Decorator(
		opts.Storage, cachesize.GetWatchCacheSizeByResource(cachesize.NetworkPolicys), &extensionsapi.NetworkPolicy{}, prefix, networkpolicy.Strategy, newListFunc)

	store := &registry.Store{
		NewFunc: func() runtime.Object { return &extensionsapi.NetworkPolicy{} },

		// NewListFunc returns an object capable of storing results of an etcd list.
		NewListFunc: newListFunc,
		// Produces a NetworkPolicy that etcd understands, to the root of the resource
		// by combining the namespace in the context with the given prefix
		KeyRootFunc: func(ctx api.Context) string {
			return registry.NamespaceKeyRootFunc(ctx, prefix)
		},
		// Produces a NetworkPolicy that etcd understands, to the resource by combining
		// the namespace in the context with the given prefix
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return registry.NamespaceKeyFunc(ctx, prefix, name)
		},
		// Retrieve the name field of a network policy
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*extensionsapi.NetworkPolicy).Name, nil
		},
		// Used to match objects based on labels/fields for list and watch
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return networkpolicy.MatchNetworkPolicy(label, field)
		},
		QualifiedResource:       extensionsapi.Resource("networkpolicies"),
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,

		// Used to validate controller creation
		CreateStrategy: networkpolicy.Strategy,

		// Used to validate controller updates
		UpdateStrategy: networkpolicy.Strategy,
		DeleteStrategy: networkpolicy.Strategy,

		Storage: storageInterface,
	}
	return &REST{store}
}
