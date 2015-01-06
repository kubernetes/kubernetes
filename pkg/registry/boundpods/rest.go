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

package boundpods

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// REST adapts a bound pods registry into apiserver's RESTStorage model.
type REST struct {
	registry Registry
}

// NewREST returns a new REST. You must use a registry created by
// NewEtcdRegistry unless you're testing.
func NewREST(registry Registry) *REST {
	return &REST{
		registry: registry,
	}
}

// New returns a new api.Event
func (*REST) New() runtime.Object {
	return &api.BoundPods{}
}

// Get returns a bound pod identified by its node name.
func (rs *REST) Get(ctx api.Context, node string) (runtime.Object, error) {
	obj, err := rs.registry.Get(ctx, node)
	if err != nil {
		return nil, err
	}
	boundpods, ok := obj.(*api.BoundPods)
	if !ok {
		return nil, fmt.Errorf("invalid object type")
	}
	return boundpods, err
}

// Watch returns BoundPods via a watch.Interface.
// It implements apiserver.ResourceWatcher.
func (rs *REST) Watch(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	return rs.registry.Watch(ctx, label, field, resourceVersion)
}

// Create returns an error: BoundPods are not mutable.
func (rs *REST) Create(ctx api.Context, obj runtime.Object) (<-chan apiserver.RESTResult, error) {
	return nil, fmt.Errorf("not allowed: 'BoundPods' objects are not mutable")
}

// Delete returns an error: BoundPods are not mutable.
func (rs *REST) Delete(ctx api.Context, id string) (<-chan apiserver.RESTResult, error) {
	return nil, fmt.Errorf("not allowed: 'BoundPods' objects are not mutable")
}

// List returns an error: BoundPods are not listable.
func (rs *REST) List(ctx api.Context, label, field labels.Selector) (runtime.Object, error) {
	return nil, fmt.Errorf("not allowed: 'BoundPods' objects are not listable")
}

// Update returns an error: BoundPods are not mutable.
func (rs *REST) Update(ctx api.Context, obj runtime.Object) (<-chan apiserver.RESTResult, error) {
	return nil, fmt.Errorf("not allowed: 'BoundPods' objects are not mutable")
}
