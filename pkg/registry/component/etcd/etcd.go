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
	"k8s.io/kubernetes/pkg/api/rest"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/registry/component"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// NewStorage returns a database backed Storage impl that stores components.
func NewStorage(s storage.Interface, connection client.ConnectionInfoGetter) rest.StandardStorage {
	prefix := "/components"
	return &etcdgeneric.Etcd{
		NewFunc:     func() runtime.Object { return &api.Component{} },
		NewListFunc: func() runtime.Object { return &api.ComponentList{} },
		KeyRootFunc: func(ctx api.Context) string {
			return prefix
		},
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return prefix + "/" + name, nil
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.Component).Name, nil
		},
		PredicateFunc: component.MatchComponent,
		EndpointName:  "components",

		CreateStrategy: component.CreateStrategy,
		UpdateStrategy: component.CreateStrategy,

		Storage: s,
	}
}
