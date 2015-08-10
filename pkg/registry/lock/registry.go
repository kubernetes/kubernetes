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

package lock

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/registry/generic"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

type registry struct {
	*etcdgeneric.Etcd
}

// NewEtcdRegistry returns a registry which will store Lock in the given etcdStorage
func NewEtcdRegistry(s storage.Interface) generic.Registry {
	prefix := "/locks"

	return registry{
		Etcd: &etcdgeneric.Etcd{
			NewFunc:      func() runtime.Object { return &api.Lock{} },
			NewListFunc:  func() runtime.Object { return &api.LockList{} },
			EndpointName: "locks",
			KeyRootFunc: func(ctx api.Context) string {
				return etcdgeneric.NamespaceKeyRootFunc(ctx, prefix)
			},
			KeyFunc: func(ctx api.Context, name string) (string, error) {
				return etcdgeneric.NamespaceKeyFunc(ctx, prefix, name)
			},
			ObjectNameFunc: func(obj runtime.Object) (string, error) {
				return obj.(*api.Lock).Name, nil
			},
			Storage: s,
			TTLFunc: func(obj runtime.Object, existin uint64, update bool) (uint64, error) {
				return obj.(*api.Lock).Spec.LeaseTime, nil
			},
		},
	}
}
