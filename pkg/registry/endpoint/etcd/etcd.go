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
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/endpoint"
	"k8s.io/kubernetes/pkg/registry/generic"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

type REST struct {
	*etcdgeneric.Etcd
}

// NewREST returns a RESTStorage object that will work against endpoints.
func NewREST(s storage.Interface, useCacher bool) *REST {
	prefix := "/services/endpoints"

	storageInterface := s
	if useCacher {
		config := storage.CacherConfig{
			CacheCapacity:  1000,
			Storage:        s,
			Type:           &api.Endpoints{},
			ResourcePrefix: prefix,
			KeyFunc: func(obj runtime.Object) (string, error) {
				return storage.NamespaceKeyFunc(prefix, obj)
			},
			NewListFunc: func() runtime.Object { return &api.EndpointsList{} },
		}
		storageInterface = storage.NewCacher(config)
	}

	store := &etcdgeneric.Etcd{
		NewFunc:     func() runtime.Object { return &api.Endpoints{} },
		NewListFunc: func() runtime.Object { return &api.EndpointsList{} },
		KeyRootFunc: func(ctx api.Context) string {
			return etcdgeneric.NamespaceKeyRootFunc(ctx, prefix)
		},
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return etcdgeneric.NamespaceKeyFunc(ctx, prefix, name)
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.Endpoints).Name, nil
		},
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return endpoint.MatchEndpoints(label, field)
		},
		EndpointName: "endpoints",

		CreateStrategy: endpoint.Strategy,
		UpdateStrategy: endpoint.Strategy,

		Storage: storageInterface,
	}
	return &REST{store}
}
