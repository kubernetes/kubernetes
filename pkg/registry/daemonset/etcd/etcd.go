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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/daemonset"
	"k8s.io/kubernetes/pkg/registry/generic"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// rest implements a RESTStorage for DaemonSets against etcd
type REST struct {
	*etcdgeneric.Etcd
}

// daemonPrefix is the location for daemons in etcd
var daemonPrefix = "/daemonsets"

// NewREST returns a RESTStorage object that will work against DaemonSets.
func NewREST(s storage.Interface) (*REST, *StatusREST) {
	store := &etcdgeneric.Etcd{
		NewFunc: func() runtime.Object { return &extensions.DaemonSet{} },

		// NewListFunc returns an object capable of storing results of an etcd list.
		NewListFunc: func() runtime.Object { return &extensions.DaemonSetList{} },
		// Produces a path that etcd understands, to the root of the resource
		// by combining the namespace in the context with the given prefix
		KeyRootFunc: func(ctx api.Context) string {
			return etcdgeneric.NamespaceKeyRootFunc(ctx, daemonPrefix)
		},
		// Produces a path that etcd understands, to the resource by combining
		// the namespace in the context with the given prefix
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return etcdgeneric.NamespaceKeyFunc(ctx, daemonPrefix, name)
		},
		// Retrieve the name field of a daemon set
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*extensions.DaemonSet).Name, nil
		},
		// Used to match objects based on labels/fields for list and watch
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return daemonset.MatchDaemonSet(label, field)
		},
		EndpointName: "daemonsets",

		// Used to validate daemon set creation
		CreateStrategy: daemonset.Strategy,

		// Used to validate daemon set updates
		UpdateStrategy: daemonset.Strategy,

		Storage: s,
	}
	statusStore := *store
	statusStore.UpdateStrategy = daemonset.StatusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}
}

// StatusREST implements the REST endpoint for changing the status of a daemonset
type StatusREST struct {
	store *etcdgeneric.Etcd
}

func (r *StatusREST) New() runtime.Object {
	return &extensions.DaemonSet{}
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	return r.store.Update(ctx, obj)
}
