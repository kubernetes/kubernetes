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

package unversioned

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

type ComponentsClientSupplier interface {
	ComponentsClient() ComponentsClient
}

type ComponentsClient interface {
	Get(name string) (result *api.Component, err error)
	Create(node *api.Component) (*api.Component, error)
	List(label labels.Selector, field fields.Selector) (*api.ComponentList, error)
	Delete(name string) error
	Update(*api.Component) (*api.Component, error)
	UpdateStatus(*api.Component) (*api.Component, error)
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
}

// componentClient implements ComponentClient
type componentClient struct {
	r *Client
}

// newComponentClient returns a new ComponentClient.
func newComponentClient(c *Client) ComponentsClient {
	return &componentClient{c}
}

// resourceName returns node's URL resource name.
func (c *componentClient) resourceName() string {
	return "components"
}

// Create registers a new component instance.
func (c *componentClient) Create(component *api.Component) (*api.Component, error) {
	result := &api.Component{}
	err := c.r.Post().Resource(c.resourceName()).Body(component).Do().Into(result)
	return result, err
}

// List takes a selector, and returns the list of registered components that match that selector.
func (c *componentClient) List(label labels.Selector, field fields.Selector) (*api.ComponentList, error) {
	result := &api.ComponentList{}
	err := c.r.Get().Resource(c.resourceName()).LabelsSelectorParam(label).FieldsSelectorParam(field).Do().Into(result)
	return result, err
}

// Get gets a registered component.
func (c *componentClient) Get(name string) (*api.Component, error) {
	result := &api.Component{}
	err := c.r.Get().Resource(c.resourceName()).Name(name).Do().Into(result)
	return result, err
}

// Delete deletes a registered component.
func (c *componentClient) Delete(name string) error {
	return c.r.Delete().Resource(c.resourceName()).Name(name).Do().Error()
}

// Update updates an existing component registration.
func (c *componentClient) Update(component *api.Component) (*api.Component, error) {
	result := &api.Component{}
	if len(component.ResourceVersion) == 0 {
		err := fmt.Errorf("invalid update object, missing resource version: %v", component)
		return nil, err
	}
	err := c.r.Put().Resource(c.resourceName()).Name(component.Name).Body(component).Do().Into(result)
	return result, err
}

// UpdateStatus updates the status of an existing component registration.
func (c *componentClient) UpdateStatus(component *api.Component) (*api.Component, error) {
	result := &api.Component{}
	if len(component.ResourceVersion) == 0 {
		err := fmt.Errorf("invalid update object, missing resource version: %v", component)
		return nil, err
	}
	err := c.r.Put().Resource(c.resourceName()).Name(component.Name).SubResource("status").Body(component).Do().Into(result)
	return result, err
}

// Watch returns a watch.Interface that watches the requested componentClient.
func (c *componentClient) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		//	Namespace(api.NamespaceAll).
		Resource(c.resourceName()).
		Param("resourceVersion", resourceVersion).
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Watch()
}
