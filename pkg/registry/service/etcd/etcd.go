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
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

type REST struct {
	*etcdgeneric.Etcd
}

// NewREST returns a RESTStorage object that will work against services.
func NewREST(s storage.Interface) *REST {
	prefix := "/services/specs"
	store := &etcdgeneric.Etcd{
		NewFunc:     func() runtime.Object { return &api.Service{} },
		NewListFunc: func() runtime.Object { return &api.ServiceList{} },
		KeyRootFunc: func(ctx api.Context) string {
			return etcdgeneric.NamespaceKeyRootFunc(ctx, prefix)
		},
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return etcdgeneric.NamespaceKeyFunc(ctx, prefix, name)
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.Service).Name, nil
		},
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return MatchServices(label, field)
		},
		EndpointName: "services",

		CreateStrategy: rest.Services,
		UpdateStrategy: rest.Services,

		Storage: s,
	}
	return &REST{store}
}

// FIXME: Move it.
func MatchServices(label labels.Selector, field fields.Selector) generic.Matcher {
	return &generic.SelectionPredicate{Label: label, Field: field, GetAttrs: ServiceAttributes}
}

func ServiceAttributes(obj runtime.Object) (objLabels labels.Set, objFields fields.Set, err error) {
	service, ok := obj.(*api.Service)
	if !ok {
		return nil, nil, fmt.Errorf("invalid object type %#v", obj)
	}
	return service.Labels, fields.Set{
		"metadata.name": service.Name,
	}, nil
}
