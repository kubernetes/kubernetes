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
	"net/http"
	"net/url"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/client"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/registry/minion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

type REST struct {
	*etcdgeneric.Etcd
	connection client.ConnectionInfoGetter
}

// StatusREST implements the REST endpoint for changing the status of a pod.
type StatusREST struct {
	store *etcdgeneric.Etcd
}

func (r *StatusREST) New() runtime.Object {
	return &api.Node{}
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	return r.store.Update(ctx, obj)
}

// NewStorage returns a RESTStorage object that will work against nodes.
func NewStorage(s storage.Interface, connection client.ConnectionInfoGetter) (*REST, *StatusREST) {
	prefix := "/minions"
	store := &etcdgeneric.Etcd{
		NewFunc:     func() runtime.Object { return &api.Node{} },
		NewListFunc: func() runtime.Object { return &api.NodeList{} },
		KeyRootFunc: func(ctx api.Context) string {
			return prefix
		},
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return prefix + "/" + name, nil
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.Node).Name, nil
		},
		PredicateFunc: minion.MatchNode,
		EndpointName:  "node",

		CreateStrategy: minion.Strategy,
		UpdateStrategy: minion.Strategy,

		Storage: s,
	}

	statusStore := *store
	statusStore.UpdateStrategy = minion.StatusStrategy

	return &REST{store, connection}, &StatusREST{store: &statusStore}
}

// Implement Redirector.
var _ = rest.Redirector(&REST{})

// ResourceLocation returns a URL to which one can send traffic for the specified minion.
func (r *REST) ResourceLocation(ctx api.Context, id string) (*url.URL, http.RoundTripper, error) {
	return minion.ResourceLocation(r, r.connection, ctx, id)
}
