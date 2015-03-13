/*
Copyright 2015 Google Inc. All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	etcdgeneric "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/resourcequota"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// rest implements a RESTStorage for resourcequotas against etcd
type REST struct {
	store *etcdgeneric.Etcd
}

// NewREST returns a RESTStorage object that will work against ResourceQuota objects.
func NewREST(h tools.EtcdHelper) (*REST, *StatusREST) {
	prefix := "/registry/resourcequotas"
	store := &etcdgeneric.Etcd{
		NewFunc:     func() runtime.Object { return &api.ResourceQuota{} },
		NewListFunc: func() runtime.Object { return &api.ResourceQuotaList{} },
		KeyRootFunc: func(ctx api.Context) string {
			return etcdgeneric.NamespaceKeyRootFunc(ctx, prefix)
		},
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return etcdgeneric.NamespaceKeyFunc(ctx, prefix, name)
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.ResourceQuota).Name, nil
		},
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return resourcequota.MatchResourceQuota(label, field)
		},
		EndpointName: "resourcequotas",

		Helper: h,
	}

	store.CreateStrategy = resourcequota.Strategy
	store.UpdateStrategy = resourcequota.Strategy
	store.ReturnDeletedObject = true

	statusStore := *store
	statusStore.UpdateStrategy = resourcequota.StatusStrategy

	return &REST{store: store}, &StatusREST{store: &statusStore}
}

// New returns a new object
func (r *REST) New() runtime.Object {
	return r.store.NewFunc()
}

// NewList returns a new list object
func (r *REST) NewList() runtime.Object {
	return r.store.NewListFunc()
}

// List obtains a list of resourcequotas with labels that match selector.
func (r *REST) List(ctx api.Context, label labels.Selector, field fields.Selector) (runtime.Object, error) {
	return r.store.List(ctx, label, field)
}

// Watch begins watching for new, changed, or deleted resourcequotas.
func (r *REST) Watch(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return r.store.Watch(ctx, label, field, resourceVersion)
}

// Get gets a specific resourcequota specified by its ID.
func (r *REST) Get(ctx api.Context, name string) (runtime.Object, error) {
	return r.store.Get(ctx, name)
}

// Create creates a resourcequota based on a specification.
func (r *REST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	return r.store.Create(ctx, obj)
}

// Update changes a resourcequota specification.
func (r *REST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	return r.store.Update(ctx, obj)
}

// Delete deletes an existing resourcequota specified by its name.
func (r *REST) Delete(ctx api.Context, name string) (runtime.Object, error) {
	return r.store.Delete(ctx, name)
}

// StatusREST implements the REST endpoint for changing the status of a resourcequota.
type StatusREST struct {
	store *etcdgeneric.Etcd
}

func (r *StatusREST) New() runtime.Object {
	return &api.ResourceQuota{}
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	return r.store.Update(ctx, obj)
}
