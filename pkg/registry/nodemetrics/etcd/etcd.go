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
	"k8s.io/kubernetes/pkg/apis/experimental"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/registry/nodemetrics"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// REST implements a RESTStorage for node metrics against etcd.
type REST struct {
	*etcdgeneric.Etcd
}

// NewREST returns a RESTStorage object that will work against node metrics.
func NewREST(s storage.Interface) *REST {
	prefix := "/nodemetrics"
	store := &etcdgeneric.Etcd{
		NewFunc: func() runtime.Object {
			return &experimental.DerivedNodeMetrics{}
		},
		NewListFunc: func() runtime.Object {
			return &experimental.DerivedNodeMetricsList{}
		},
		KeyRootFunc: func(ctx api.Context) string {
			return prefix
		},
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return prefix + "/" + name, nil
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*experimental.DerivedNodeMetrics).Name, nil
		},
		PredicateFunc:  nodemetrics.MatchDerivedNodeMetrics,
		EndpointName:   "nodemetrics",
		CreateStrategy: nodemetrics.Strategy,
		UpdateStrategy: nodemetrics.Strategy,
		Storage:        s,
	}

	return &REST{store}
}
