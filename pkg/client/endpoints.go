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

package client

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// EndpointsNamespacer has methods to work with Endpoint resources in a namespace
type EndpointsNamespacer interface {
	Endpoint(namespace string) EndpointsInterface
}

// EndpointsInterface has methods to work with Endpoints resources
type EndpointsInterface interface {
	Create(endpoints *api.Endpoints) (*api.Endpoints, error)
	List(selector labels.Selector) (*api.EndpointsList, error)
	Get(id string) (*api.Endpoints, error)
	Update(endpoints *api.Endpoints) (*api.Endpoints, error)
	Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error)
}

// endpoints implements EndpointsNamespacer interface
type endpoints struct {
	r  *Client
	ns string
}

// newEndpoints returns a endpoints
func newEndpoints(c *Client, namespace string) *endpoints {
	return &endpoints{c, namespace}
}

// Create creates a new endpoint.
func (c *endpoints) Create(endpoints *api.Endpoints) (*api.Endpoints, error) {
	result := &api.Endpoints{}
	err := c.r.Post().Namespace(c.ns).Path("endpoints").Body(endpoints).Do().Into(result)
	return result, err
}

// List takes a selector, and returns the list of endpoints that match that selector
func (c *endpoints) List(selector labels.Selector) (result *api.EndpointsList, err error) {
	result = &api.EndpointsList{}
	err = c.r.Get().Namespace(c.ns).Path("endpoints").SelectorParam("labels", selector).Do().Into(result)
	return
}

// Get returns information about the endpoints for a particular service.
func (c *endpoints) Get(id string) (result *api.Endpoints, err error) {
	result = &api.Endpoints{}
	err = c.r.Get().Namespace(c.ns).Path("endpoints").Path(id).Do().Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested endpoints for a service.
func (c *endpoints) Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	return c.r.Get().
		Namespace(c.ns).
		Path("watch").
		Path("endpoints").
		Param("resourceVersion", resourceVersion).
		SelectorParam("labels", label).
		SelectorParam("fields", field).
		Watch()
}

func (c *endpoints) Update(endpoints *api.Endpoints) (*api.Endpoints, error) {
	result := &api.Endpoints{}
	if len(endpoints.ResourceVersion) == 0 {
		return nil, fmt.Errorf("invalid update object, missing resource version: %v", endpoints)
	}
	err := c.r.Put().
		Namespace(c.ns).
		Path("endpoints").
		Path(endpoints.Name).
		Body(endpoints).
		Do().
		Into(result)
	return result, err
}
