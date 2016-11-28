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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/generic"
	genericregistry "k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/registry/rbac/clusterrolebinding"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// REST implements a RESTStorage for ClusterRoleBinding against etcd
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against ClusterRoleBinding objects.
func NewREST(opts generic.RESTOptions) *REST {
	prefix := "/" + opts.ResourcePrefix

	newListFunc := func() runtime.Object { return &rbac.ClusterRoleBindingList{} }
	storageInterface, dFunc := opts.Decorator(
		opts.StorageConfig,
		cachesize.GetWatchCacheSizeByResource(cachesize.ClusterRoleBindings),
		&rbac.ClusterRoleBinding{},
		prefix,
		clusterrolebinding.Strategy,
		newListFunc,
		clusterrolebinding.GetAttrs,
		storage.NoTriggerPublisher,
	)

	store := &genericregistry.Store{
		NewFunc:     func() runtime.Object { return &rbac.ClusterRoleBinding{} },
		NewListFunc: newListFunc,
		KeyRootFunc: func(ctx api.Context) string {
			return genericregistry.NamespaceKeyRootFunc(ctx, prefix)
		},
		KeyFunc: func(ctx api.Context, id string) (string, error) {
			return genericregistry.NoNamespaceKeyFunc(ctx, prefix, id)
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*rbac.ClusterRoleBinding).Name, nil
		},
		PredicateFunc:           clusterrolebinding.Matcher,
		QualifiedResource:       rbac.Resource("clusterrolebindings"),
		EnableGarbageCollection: opts.EnableGarbageCollection,
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,

		CreateStrategy: clusterrolebinding.Strategy,
		UpdateStrategy: clusterrolebinding.Strategy,
		DeleteStrategy: clusterrolebinding.Strategy,

		Storage:     storageInterface,
		DestroyFunc: dFunc,
	}

	return &REST{store}
}
