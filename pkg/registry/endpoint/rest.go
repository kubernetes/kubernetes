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

package endpoint

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// REST adapts endpoints into apiserver's RESTStorage model.
type REST struct {
	registry Registry
}

// NewREST returns a new apiserver.RESTStorage implementation for endpoints
func NewREST(registry Registry) *REST {
	return &REST{
		registry: registry,
	}
}

// Get satisfies the RESTStorage interface.
func (rs *REST) Get(ctx api.Context, id string) (runtime.Object, error) {
	return rs.registry.GetEndpoints(ctx, id)
}

// List satisfies the RESTStorage interface.
func (rs *REST) List(ctx api.Context, label, field labels.Selector) (runtime.Object, error) {
	if !label.Empty() || !field.Empty() {
		return nil, errors.NewBadRequest("label/field selectors are not supported on endpoints")
	}
	return rs.registry.ListEndpoints(ctx)
}

// Watch returns Endpoint events via a watch.Interface.
// It implements apiserver.ResourceWatcher.
func (rs *REST) Watch(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	return rs.registry.WatchEndpoints(ctx, label, field, resourceVersion)
}

// Create satisfies the RESTStorage interface.
func (rs *REST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	endpoints, ok := obj.(*api.Endpoints)
	if !ok {
		return nil, fmt.Errorf("not an endpoints: %#v", obj)
	}
	if len(endpoints.Name) == 0 {
		return nil, fmt.Errorf("id is required: %#v", obj)
	}
	if !api.ValidNamespace(ctx, &endpoints.ObjectMeta) {
		return nil, errors.NewConflict("endpoints", endpoints.Namespace, fmt.Errorf("Endpoints.Namespace does not match the provided context"))
	}
	api.FillObjectMetaSystemFields(ctx, &endpoints.ObjectMeta)

	err := rs.registry.UpdateEndpoints(ctx, endpoints)
	if err != nil {
		return nil, err
	}
	return rs.registry.GetEndpoints(ctx, endpoints.Name)
}

// Update satisfies the RESTStorage interface.
func (rs *REST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	endpoints, ok := obj.(*api.Endpoints)
	if !ok {
		return nil, false, fmt.Errorf("not an endpoints: %#v", obj)
	}
	err := rs.registry.UpdateEndpoints(ctx, endpoints)
	if err != nil {
		return nil, false, err
	}
	out, err := rs.registry.GetEndpoints(ctx, endpoints.Name)
	return out, false, err
}

// New implements the RESTStorage interface.
func (rs REST) New() runtime.Object {
	return &api.Endpoints{}
}

func (*REST) NewList() runtime.Object {
	return &api.EndpointsList{}
}
