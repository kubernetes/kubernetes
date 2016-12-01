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
	extensionsapi "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/registry/extensions/networkpolicy"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/util/restoptions"
)

// rest implements a RESTStorage for network policies against etcd
type REST struct {
	*registry.Store
}

// NewREST returns a RESTStorage object that will work against network policies.
func NewREST(optsGetter genericapiserver.RESTOptionsGetter) *REST {
	store := &registry.Store{
		NewFunc:     func() runtime.Object { return &extensionsapi.NetworkPolicy{} },
		NewListFunc: func() runtime.Object { return &extensionsapi.NetworkPolicyList{} },
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*extensionsapi.NetworkPolicy).Name, nil
		},
		PredicateFunc:     networkpolicy.MatchNetworkPolicy,
		QualifiedResource: extensionsapi.Resource("networkpolicies"),

		CreateStrategy: networkpolicy.Strategy,
		UpdateStrategy: networkpolicy.Strategy,
		DeleteStrategy: networkpolicy.Strategy,
	}
	restoptions.ApplyOptions(optsGetter, store, storage.NoTriggerPublisher)
	return &REST{store}
}
