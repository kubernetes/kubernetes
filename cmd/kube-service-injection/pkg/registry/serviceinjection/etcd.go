/*
Copyright 2017 The Kubernetes Authors.

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

package serviceinjection

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/cmd/kube-service-injection/pkg/apis/serviceinjection"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/genericapiserver/registry/generic"
	genericregistry "k8s.io/kubernetes/pkg/genericapiserver/registry/generic/registry"
)

// rest implements a RESTStorage for API services against etcd
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against API services.
func NewREST(optsGetter generic.RESTOptionsGetter) *REST {
	store := &genericregistry.Store{
		Copier:      api.Scheme,
		NewFunc:     func() runtime.Object { return &serviceinjection.ServiceInjection{} },
		NewListFunc: func() runtime.Object { return &serviceinjection.ServiceInjectionList{} },
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*serviceinjection.ServiceInjection).Name, nil
		},
		PredicateFunc:     MatchServiceInjection,
		QualifiedResource: serviceinjection.Resource("apiservices"),

		CreateStrategy: Strategy,
		UpdateStrategy: Strategy,
		DeleteStrategy: Strategy,
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: GetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}
	return &REST{store}
}
