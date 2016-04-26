/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/federation/registry/cluster"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/registry/generic"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/runtime"
)

type REST struct {
	*etcdgeneric.Etcd
}

type StatusREST struct {
	store *etcdgeneric.Etcd
}

func (r *StatusREST) New() runtime.Object {
	return &federation.Cluster{}
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	return r.store.Update(ctx, obj)
}

// NewREST returns a RESTStorage object that will work against clusters.
func NewREST(opts generic.RESTOptions) (*REST, *StatusREST) {
	prefix := "/clusters"

	newListFunc := func() runtime.Object { return &federation.ClusterList{} }
	storageInterface := opts.Decorator(
		opts.Storage, 100, &federation.Cluster{}, prefix, cluster.Strategy, newListFunc)

	store := &etcdgeneric.Etcd{
		NewFunc:     func() runtime.Object { return &federation.Cluster{} },
		NewListFunc: newListFunc,
		KeyRootFunc: func(ctx api.Context) string {
			return prefix
		},
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return etcdgeneric.NoNamespaceKeyFunc(ctx, prefix, name)
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*federation.Cluster).Name, nil
		},
		PredicateFunc:           cluster.MatchCluster,
		QualifiedResource:       federation.Resource("clusters"),
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,

		CreateStrategy: cluster.Strategy,
		UpdateStrategy: cluster.Strategy,
		DeleteStrategy: cluster.Strategy,

		ReturnDeletedObject: true,

		Storage: storageInterface,
	}

	statusStore := *store
	statusStore.UpdateStrategy = cluster.StatusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}
}
