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
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/genericapiserver"
	ingress "k8s.io/kubernetes/pkg/registry/extensions/ingress"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/util/restoptions"
)

// rest implements a RESTStorage for replication controllers against etcd
type REST struct {
	*registry.Store
}

// NewREST returns a RESTStorage object that will work against replication controllers.
func NewREST(optsGetter genericapiserver.RESTOptionsGetter) (*REST, *StatusREST) {
	store := &registry.Store{
		NewFunc:     func() runtime.Object { return &extensions.Ingress{} },
		NewListFunc: func() runtime.Object { return &extensions.IngressList{} },
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*extensions.Ingress).Name, nil
		},
		PredicateFunc:     ingress.MatchIngress,
		QualifiedResource: extensions.Resource("ingresses"),

		CreateStrategy: ingress.Strategy,
		UpdateStrategy: ingress.Strategy,
		DeleteStrategy: ingress.Strategy,
	}
	restoptions.ApplyOptions(optsGetter, store, storage.NoTriggerPublisher)

	statusStore := *store
	statusStore.UpdateStrategy = ingress.StatusStrategy
	return &REST{store}, &StatusREST{store: &statusStore}
}

// StatusREST implements the REST endpoint for changing the status of an ingress
type StatusREST struct {
	store *registry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &extensions.Ingress{}
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx api.Context, name string) (runtime.Object, error) {
	return r.store.Get(ctx, name)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}
