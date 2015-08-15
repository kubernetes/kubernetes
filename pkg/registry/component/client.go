/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package component

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	kclient "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// client implements client.ComponentsClient by wrapping a component.Registry
type client struct {
	store       rest.StandardStorage
	statusStore rest.Patcher
}

// NewRegistry returns a new ComponentsClient backed by the provided store and status store. Any mismatched types will panic.
func NewClient(store rest.StandardStorage, statusStore rest.Patcher) kclient.ComponentsClient {
	return &client{
		store:       store,
		statusStore: statusStore,
	}
}

// Create registers a new component instance.
func (c *client) Create(state *api.Component) (*api.Component, error) {
	newState, err := c.store.Create(api.NewContext(), state)
	if err != nil {
		return nil, fmt.Errorf("failed to create component: %v", err)
	}
	return newState.(*api.Component), nil
}

// List takes a selector, and returns the list of registered components that match that selector.
func (c *client) List(label labels.Selector, field fields.Selector) (*api.ComponentList, error) {
	list, err := c.store.List(api.NewContext(), label, field)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve component list: %v", err)
	}
	return list.(*api.ComponentList), nil
}

// Get gets a registered component.
func (c *client) Get(name string) (*api.Component, error) {
	state, err := c.store.Get(api.NewContext(), name)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve component by name (%s): %v", name, err)
	}
	return state.(*api.Component), nil
}

// Delete deletes a registered component.
func (c *client) Delete(name string) error {
	options := &api.DeleteOptions{}
	_, err := c.store.Delete(api.NewContext(), name, options)
	if err != nil {
		return fmt.Errorf("failed to delete component: %v", err)
	}
	return nil
}

// Update updates an existing component registration.
func (c *client) Update(state *api.Component) (*api.Component, error) {
	newState, _, err := c.store.Update(api.NewContext(), state)
	if err != nil {
		return nil, fmt.Errorf("failed to update component: %v", err)
	}
	return newState.(*api.Component), nil
}

// UpdateStatus updates the status of an existing component registration.
func (c *client) UpdateStatus(state *api.Component) (*api.Component, error) {
	newState, _, err := c.statusStore.Update(api.NewContext(), state)
	if err != nil {
		return nil, fmt.Errorf("failed to update component: %v", err)
	}
	return newState.(*api.Component), nil
}

// Watch returns a watch.Interface that watches the requested componentClient.
func (c *client) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	iface, err := c.store.Watch(api.NewContext(), label, field, resourceVersion)
	if err != nil {
		return nil, fmt.Errorf("failed to watch component: %v", err)
	}
	return iface, nil
}
