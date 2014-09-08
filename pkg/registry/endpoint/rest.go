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
	"errors"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
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
func (rs *REST) Get(id string) (runtime.Object, error) {
	return rs.registry.GetEndpoints(id)
}

// List satisfies the RESTStorage interface.
func (rs *REST) List(selector labels.Selector) (runtime.Object, error) {
	if !selector.Empty() {
		return nil, errors.New("label selectors are not supported on endpoints")
	}
	return rs.registry.ListEndpoints()
}

// Watch returns Endpoint events via a watch.Interface.
// It implements apiserver.ResourceWatcher.
func (rs *REST) Watch(label, field labels.Selector, resourceVersion uint64) (watch.Interface, error) {
	return rs.registry.WatchEndpoints(label, field, resourceVersion)
}

// Create satisfies the RESTStorage interface but is unimplemented.
func (rs *REST) Create(obj runtime.Object) (<-chan runtime.Object, error) {
	return nil, errors.New("unimplemented")
}

// Update satisfies the RESTStorage interface but is unimplemented.
func (rs *REST) Update(obj runtime.Object) (<-chan runtime.Object, error) {
	return nil, errors.New("unimplemented")
}

// Delete satisfies the RESTStorage interface but is unimplemented.
func (rs *REST) Delete(id string) (<-chan runtime.Object, error) {
	return nil, errors.New("unimplemented")
}

// New implements the RESTStorage interface.
func (rs REST) New() runtime.Object {
	return &api.Endpoints{}
}
