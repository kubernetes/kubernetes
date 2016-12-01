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
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/registry/rbac/role"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/util/restoptions"
)

// REST implements a RESTStorage for Role against etcd
type REST struct {
	*registry.Store
}

// NewREST returns a RESTStorage object that will work against Role objects.
func NewREST(optsGetter genericapiserver.RESTOptionsGetter) *REST {
	store := &registry.Store{
		NewFunc:     func() runtime.Object { return &rbac.Role{} },
		NewListFunc: func() runtime.Object { return &rbac.RoleList{} },
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*rbac.Role).Name, nil
		},
		PredicateFunc:     role.Matcher,
		QualifiedResource: rbac.Resource("roles"),

		CreateStrategy: role.Strategy,
		UpdateStrategy: role.Strategy,
		DeleteStrategy: role.Strategy,
	}
	restoptions.ApplyOptions(optsGetter, store, storage.NoTriggerPublisher)

	return &REST{store}
}
