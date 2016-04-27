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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/registry/template"
	"k8s.io/kubernetes/pkg/runtime"
)

// REST implements a RESTStorage for templates against etcd
type REST struct {
	*registry.Store
}

// NewREST returns a RESTStorage object that will work against replication controllers.
func NewREST(opts generic.RESTOptions) *REST {
	prefix := "/template"

	newListFunc := func() runtime.Object { return &extensions.TemplateList{} }
	storageInterface := opts.Decorator(
		opts.Storage, cachesize.GetWatchCacheSizeByResource(cachesize.Templates), &extensions.Template{}, prefix, template.Strategy, newListFunc)

	store := &registry.Store{
		NewFunc: func() runtime.Object { return &extensions.Template{} },

		// NewListFunc returns an object capable of storing results of an etcd list.
		NewListFunc: newListFunc,
		// Produces a templates that etcd understands, to the root of the resource
		// by combining the namespace in the context with the given prefix
		KeyRootFunc: func(ctx api.Context) string {
			return registry.NamespaceKeyRootFunc(ctx, prefix)
		},
		// Produces a templates that etcd understands, to the resource by combining
		// the namespace in the context with the given prefix
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return registry.NamespaceKeyFunc(ctx, prefix, name)
		},
		// Retrieve the name field of a template
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*extensions.Template).Name, nil
		},
		// Used to match objects based on labels/fields for list and watch
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return template.Matcher(label, field)
		},
		QualifiedResource:       extensions.Resource("templates"),
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,

		// Used to validate template creation
		CreateStrategy: template.Strategy,

		// Used to validate template updates
		UpdateStrategy: template.Strategy,

		// Used to validate template deletion
		DeleteStrategy: template.Strategy,

		Storage: storageInterface,
	}
	return &REST{store}
}

type ProcessREST struct {
	store *registry.Store
}

func (r *ProcessREST) New() runtime.Object {
	return &extensions.Template{}
}

func (r *ProcessREST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	return &api.List{}, true, nil
}
