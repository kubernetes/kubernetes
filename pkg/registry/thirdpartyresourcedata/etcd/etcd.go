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
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/registry/thirdpartyresourcedata"
	"k8s.io/kubernetes/pkg/runtime"
)

// REST implements a RESTStorage for ThirdPartyResourceDatas against etcd
type REST struct {
	*registry.Store
	kind string
}

// NewREST returns a registry which will store ThirdPartyResourceData in the given helper
func NewREST(opts generic.RESTOptions, group, kind string) *REST {
	prefix := "/ThirdPartyResourceData/" + group + "/" + strings.ToLower(kind) + "s"

	// We explicitly do NOT do any decoration here yet.
	storageInterface := generic.NewRawStorage(opts.StorageConfig)

	store := &registry.Store{
		NewFunc:     func() runtime.Object { return &extensions.ThirdPartyResourceData{} },
		NewListFunc: func() runtime.Object { return &extensions.ThirdPartyResourceDataList{} },
		KeyRootFunc: func(ctx api.Context) string {
			return registry.NamespaceKeyRootFunc(ctx, prefix)
		},
		KeyFunc: func(ctx api.Context, id string) (string, error) {
			return registry.NamespaceKeyFunc(ctx, prefix, id)
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*extensions.ThirdPartyResourceData).Name, nil
		},
		PredicateFunc:           thirdpartyresourcedata.Matcher,
		QualifiedResource:       extensions.Resource("thirdpartyresourcedatas"),
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,
		CreateStrategy:          thirdpartyresourcedata.Strategy,
		UpdateStrategy:          thirdpartyresourcedata.Strategy,
		DeleteStrategy:          thirdpartyresourcedata.Strategy,

		Storage: storageInterface,
	}

	return &REST{
		Store: store,
		kind:  kind,
	}
}

// Implements the rest.KindProvider interface
func (r *REST) Kind() string {
	return r.kind
}
