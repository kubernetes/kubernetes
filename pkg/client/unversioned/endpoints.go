/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/watch"
)

// EndpointsNamespacer has methods to work with Endpoints resources in a namespace
type EndpointsNamespacer interface {
	Endpoints(namespace string) EndpointsInterface
}

// EndpointsInterface has methods to work with Endpoints resources
type EndpointsInterface interface {
	Create(endpoints *api.Endpoints) (*api.Endpoints, error)
	List(opts api.ListOptions) (*api.EndpointsList, error)
	Get(name string) (*api.Endpoints, error)
	Delete(name string) error
	Update(endpoints *api.Endpoints) (*api.Endpoints, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
}

// endpoints implements EndpointsInterface
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
	err := c.r.Post().Namespace(c.ns).Resource("endpoints").Body(endpoints).Do().Into(result)
	return result, err
}

// List takes a selector, and returns the list of endpoints that match that selector
func (c *endpoints) List(opts api.ListOptions) (result *api.EndpointsList, err error) {
	result = &api.EndpointsList{}
	err = c.r.Get().
		Namespace(c.ns).
		Resource("endpoints").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Get returns information about the endpoints for a particular service.
func (c *endpoints) Get(name string) (result *api.Endpoints, err error) {
	result = &api.Endpoints{}
	err = c.r.Get().Namespace(c.ns).Resource("endpoints").Name(name).Do().Into(result)
	return
}

// Delete takes the name of the endpoint, and returns an error if one occurs
func (c *endpoints) Delete(name string) error {
	return c.r.Delete().Namespace(c.ns).Resource("endpoints").Name(name).Do().Error()
}

// Watch returns a watch.Interface that watches the requested endpoints for a service.
func (c *endpoints) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("endpoints").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

func (c *endpoints) Update(endpoints *api.Endpoints) (*api.Endpoints, error) {
	result := &api.Endpoints{}
	err := c.r.Put().
		Namespace(c.ns).
		Resource("endpoints").
		Name(endpoints.Name).
		Body(endpoints).
		Do().
		Into(result)
	return result, err
}
