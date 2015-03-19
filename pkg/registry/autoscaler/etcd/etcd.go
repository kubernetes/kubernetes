/*
Copyright 2014 Google Inc. All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/autoscaler"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	etcdgeneric "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// rest implements a RESTStorage for pods against etcd
type REST struct {
	store *etcdgeneric.Etcd
}

func NewREST(h tools.EtcdHelper) *REST {
	prefix := "/registry/autoScalers"
	return &REST{
		store: &etcdgeneric.Etcd{
			NewFunc:     func() runtime.Object { return &api.AutoScaler{} },
			NewListFunc: func() runtime.Object { return &api.AutoScalerList{} },
			KeyRootFunc: func(ctx api.Context) string {
				return etcdgeneric.NamespaceKeyRootFunc(ctx, prefix)
			},
			KeyFunc: func(ctx api.Context, name string) (string, error) {
				return etcdgeneric.NamespaceKeyFunc(ctx, prefix, name)
			},
			ObjectNameFunc: func(obj runtime.Object) (string, error) {
				return obj.(*api.AutoScaler).Name, nil
			},
			PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
				return autoscaler.MatchAutoScaler(label, field)
			},
			CreateStrategy: autoscaler.AutoScalers,
			UpdateStrategy: autoscaler.AutoScalers,
			EndpointName:   "autoScalers",

			Helper: h,
		},
	}
}

// New returns a new object
func (r *REST) New() runtime.Object {
	return r.store.NewFunc()
}

// NewList returns a new list object
func (r *REST) NewList() runtime.Object {
	return r.store.NewListFunc()
}

// List obtains a list of autoscalers with labels that match selector.
func (r *REST) List(ctx api.Context, label labels.Selector, field fields.Selector) (runtime.Object, error) {
	return r.store.List(ctx, label, field)
}

// Watch begins watching for new, changed, or deleted autoscalers.
func (r *REST) Watch(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return r.store.Watch(ctx, label, field, resourceVersion)
}

// Get gets a specific autoscalers specified by its ID.
func (r *REST) Get(ctx api.Context, name string) (runtime.Object, error) {
	return r.store.Get(ctx, name)
}

// Create creates a autoscalers based on a specification.
func (r *REST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	return r.store.Create(ctx, obj)
}

// Update changes a autoscalers specification.
func (r *REST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	return r.store.Update(ctx, obj)
}

// Delete deletes an existing autoscalers specified by its ID.
func (r *REST) Delete(ctx api.Context, name string) (runtime.Object, error) {
	return r.store.Delete(ctx, name)
}
