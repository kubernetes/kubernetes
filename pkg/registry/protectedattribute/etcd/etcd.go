/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/registry/protectedattribute"
	"k8s.io/kubernetes/pkg/runtime"
)

type REST struct {
	*registry.Store
}

func NewREST(opts generic.RESTOptions) *REST {
	prefix := "/protectedattributes"

	newListFunc := func() runtime.Object {
		return &rbac.ProtectedAttributeList{}
	}
	storageInterface := opts.Decorator(
		opts.Storage,
		cachesize.GetWatchCacheSizeByResource(cachesize.ProtectedAttributes),
		&rbac.ProtectedAttribute{},
		prefix,
		protectedattribute.Strategy,
		newListFunc,
	)

	store := &registry.Store{
		NewFunc:     func() runtime.Object { return &rbac.ProtectedAttribute{} },
		NewListFunc: newListFunc,
		KeyRootFunc: func(ctx api.Context) string {
			return registry.NamespaceKeyRootFunc(ctx, prefix)
		},
		KeyFunc: func(ctx api.Context, id string) (string, error) {
			return registry.NamespaceKeyFunc(ctx, prefix, id)
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*rbac.ProtectedAttribute).Name, nil
		},
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return protectedattribute.Matcher(label, field)
		},
		QualifiedResource:       rbac.Resource("protectedattributes"),
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,

		CreateStrategy: protectedattribute.Strategy,
		UpdateStrategy: protectedattribute.Strategy,
		DeleteStrategy: protectedattribute.Strategy,

		Storage: storageInterface,
	}

	return &REST{store}
}
