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

package etcd

import (
	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/federation/registry/cluster"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/util/restoptions"
)

type REST struct {
	*registry.Store
}

type StatusREST struct {
	store *registry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &federation.Cluster{}
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}

// NewREST returns a RESTStorage object that will work against clusters.
func NewREST(optsGetter genericapiserver.RESTOptionsGetter) (*REST, *StatusREST) {
	store := &registry.Store{
		NewFunc:     func() runtime.Object { return &federation.Cluster{} },
		NewListFunc: func() runtime.Object { return &federation.ClusterList{} },
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*federation.Cluster).Name, nil
		},
		PredicateFunc:     cluster.MatchCluster,
		QualifiedResource: federation.Resource("clusters"),

		CreateStrategy:      cluster.Strategy,
		UpdateStrategy:      cluster.Strategy,
		DeleteStrategy:      cluster.Strategy,
		ReturnDeletedObject: true,
	}
	restoptions.ApplyOptions(optsGetter, store, storage.NoTriggerPublisher)

	statusStore := *store
	statusStore.UpdateStrategy = cluster.StatusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}
}
