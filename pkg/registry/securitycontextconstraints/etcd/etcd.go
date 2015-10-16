/*
Copyright 2014 The Kubernetes Authors.

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
	"path"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/registry/securitycontextconstraints"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// REST implements a RESTStorage for security context constraints against etcd
type REST struct {
	*registry.Store
}

const Prefix = "/securitycontextconstraints"

// NewStorage returns a RESTStorage object that will work against security context constraints objects.
func NewStorage(opts generic.RESTOptions) *REST {

	newListFunc := func() runtime.Object { return &api.SecurityContextConstraintsList{} }

	storageInterface, _ := opts.Decorator(
		opts.StorageConfig,
		cachesize.GetWatchCacheSizeByResource(cachesize.SecurityContextConstraints),
		&api.SecurityContextConstraints{},
		Prefix,
		securitycontextconstraints.Strategy,
		newListFunc,
		storage.NoTriggerPublisher,
	)

	store := &registry.Store{
		NewFunc:     func() runtime.Object { return &api.SecurityContextConstraints{} },
		NewListFunc: newListFunc,
		KeyRootFunc: func(ctx api.Context) string {
			return Prefix
		},
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return path.Join(Prefix, name), nil
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.SecurityContextConstraints).Name, nil
		},
		PredicateFunc:     securitycontextconstraints.Matcher,
		QualifiedResource: api.Resource("securitycontextconstraints"),

		CreateStrategy:      securitycontextconstraints.Strategy,
		UpdateStrategy:      securitycontextconstraints.Strategy,
		ReturnDeletedObject: true,
		Storage:             storageInterface,
	}
	return &REST{store}
}
