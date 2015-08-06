/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/registry/generic"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/registry/serviceaccount"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// REST implements a RESTStorage for service accounts against etcd
type REST struct {
	*etcdgeneric.Etcd
}

const Prefix = "/serviceaccounts"

// NewStorage returns a RESTStorage object that will work against service accounts objects.
func NewStorage(s storage.Interface) *REST {
	store := &etcdgeneric.Etcd{
		NewFunc:     func() runtime.Object { return &api.ServiceAccount{} },
		NewListFunc: func() runtime.Object { return &api.ServiceAccountList{} },
		KeyRootFunc: func(ctx api.Context) string {
			return etcdgeneric.NamespaceKeyRootFunc(ctx, Prefix)
		},
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return etcdgeneric.NamespaceKeyFunc(ctx, Prefix, name)
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.ServiceAccount).Name, nil
		},
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return serviceaccount.Matcher(label, field)
		},
		EndpointName: "serviceaccounts",

		Storage: s,
	}

	store.CreateStrategy = serviceaccount.Strategy
	store.UpdateStrategy = serviceaccount.Strategy
	store.ReturnDeletedObject = true

	return &REST{store}
}
