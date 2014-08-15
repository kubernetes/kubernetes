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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// BindingStorage implements the RESTStorage interface. When bindings are written, it
// changes the location of the affected pods. This information is eventually reflected
// in the pod's CurrentState.Host field.
type BindingStorage struct {
	registry Registry
}

// NewBindingStorage makes a new BindingStorage backed by the given bindingRegistry.
func NewBindingStorage(bindingRegistry Registry) *BindingStorage {
	return &BindingStorage{
		registry: bindingRegistry,
	}
}

// List returns an error because bindings are write-only objects.
func (*BindingStorage) List(selector labels.Selector) (interface{}, error) {
	return nil, apiserver.NewNotFoundErr("binding", "list")
}

// Get returns an error because bindings are write-only objects.
func (*BindingStorage) Get(id string) (interface{}, error) {
	return nil, apiserver.NewNotFoundErr("binding", id)
}

// Delete returns an error because bindings are write-only objects.
func (*BindingStorage) Delete(id string) (<-chan interface{}, error) {
	return nil, apiserver.NewNotFoundErr("binding", id)
}

// New returns a new binding object fit for having data unmarshalled into it.
func (*BindingStorage) New() interface{} {
	return &api.Binding{}
}

// Create attempts to make the assignment indicated by the binding it recieves.
func (b *BindingStorage) Create(obj interface{}) (<-chan interface{}, error) {
	binding, ok := obj.(*api.Binding)
	if !ok {
		return nil, fmt.Errorf("incorrect type: %#v", obj)
	}
	return apiserver.MakeAsync(func() (interface{}, error) {
		if err := b.registry.ApplyBinding(binding); err != nil {
			return nil, err
		}
		return &api.Status{Status: api.StatusSuccess}, nil
	}), nil
}

// Update returns an error-- this object may not be updated.
func (b *BindingStorage) Update(obj interface{}) (<-chan interface{}, error) {
	return nil, fmt.Errorf("Bindings may not be changed.")
}
