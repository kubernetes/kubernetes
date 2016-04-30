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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/registry/thirdpartyresource"
	"k8s.io/kubernetes/pkg/runtime"
)

// REST implements a RESTStorage for ThirdPartyResources against etcd
type REST struct {
	*registry.Store
}

// NewREST returns a registry which will store ThirdPartyResource in the given helper
func NewREST(opts generic.RESTOptions) *REST {
	prefix := "/thirdpartyresources"

	// We explicitly do NOT do any decoration here yet.
	storageInterface := opts.Storage

	store := &registry.Store{
		NewFunc:     func() runtime.Object { return &extensions.ThirdPartyResource{} },
		NewListFunc: func() runtime.Object { return &extensions.ThirdPartyResourceList{} },
		KeyRootFunc: func(ctx api.Context) string {
			return prefix
		},
		KeyFunc: func(ctx api.Context, id string) (string, error) {
			return registry.NoNamespaceKeyFunc(ctx, prefix, id)
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*extensions.ThirdPartyResource).Name, nil
		},
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return thirdpartyresource.Matcher(label, field)
		},
		QualifiedResource:       extensions.Resource("thirdpartyresources"),
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,
		CreateStrategy:          thirdpartyresource.Strategy,
		UpdateStrategy:          thirdpartyresource.Strategy,
		DeleteStrategy:          thirdpartyresource.Strategy,

		Storage: storageInterface,
	}

	return &REST{store}
}
