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

package binding

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// REST implements the RESTStorage interface for bindings. When bindings are written, it
// changes the location of the affected pods. This information is eventually reflected
// in the pod's CurrentState.Host field.
type REST struct {
	registry Registry
}

// NewREST creates a new REST backed by the given bindingRegistry.
func NewREST(bindingRegistry Registry) *REST {
	return &REST{
		registry: bindingRegistry,
	}
}

// List returns an error because bindings are write-only objects.
func (*REST) List(ctx api.Context, label, field labels.Selector) (runtime.Object, error) {
	return nil, errors.NewNotFound("binding", "list")
}

// Get returns an error because bindings are write-only objects.
func (*REST) Get(ctx api.Context, id string) (runtime.Object, error) {
	return nil, errors.NewNotFound("binding", id)
}

// Delete returns an error because bindings are write-only objects.
func (*REST) Delete(ctx api.Context, id string) (<-chan apiserver.RESTResult, error) {
	return nil, errors.NewNotFound("binding", id)
}

// New returns a new binding object fit for having data unmarshalled into it.
func (*REST) New() runtime.Object {
	return &api.Binding{}
}

// Create attempts to make the assignment indicated by the binding it recieves.
func (b *REST) Create(ctx api.Context, obj runtime.Object) (<-chan apiserver.RESTResult, error) {
	binding, ok := obj.(*api.Binding)
	if !ok {
		return nil, fmt.Errorf("incorrect type: %#v", obj)
	}
	return apiserver.MakeAsync(func() (runtime.Object, error) {
		if err := b.registry.ApplyBinding(ctx, binding); err != nil {
			return nil, err
		}
		return &api.Status{Status: api.StatusSuccess}, nil
	}), nil
}

// Update returns an error-- this object may not be updated.
func (b *REST) Update(ctx api.Context, obj runtime.Object) (<-chan apiserver.RESTResult, error) {
	return nil, fmt.Errorf("Bindings may not be changed.")
}
