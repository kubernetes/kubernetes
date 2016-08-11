/*
Copyright 2016 The Kubernetes Authors.

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

package v1

import (
	api "k8s.io/client-go/1.4/pkg/api"
	v1 "k8s.io/client-go/1.4/pkg/api/v1"
	watch "k8s.io/client-go/1.4/pkg/watch"
)

// EndpointsGetter has a method to return a EndpointsInterface.
// A group's client should implement this interface.
type EndpointsGetter interface {
	Endpoints(namespace string) EndpointsInterface
}

// EndpointsInterface has methods to work with Endpoints resources.
type EndpointsInterface interface {
	Create(*v1.Endpoints) (*v1.Endpoints, error)
	Update(*v1.Endpoints) (*v1.Endpoints, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*v1.Endpoints, error)
	List(opts api.ListOptions) (*v1.EndpointsList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.Endpoints, err error)
	EndpointsExpansion
}

// endpoints implements EndpointsInterface
type endpoints struct {
	client *CoreClient
	ns     string
}

// newEndpoints returns a Endpoints
func newEndpoints(c *CoreClient, namespace string) *endpoints {
	return &endpoints{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a endpoints and creates it.  Returns the server's representation of the endpoints, and an error, if there is any.
func (c *endpoints) Create(endpoints *v1.Endpoints) (result *v1.Endpoints, err error) {
	result = &v1.Endpoints{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("endpoints").
		Body(endpoints).
		Do().
		Into(result)
	return
}

// Update takes the representation of a endpoints and updates it. Returns the server's representation of the endpoints, and an error, if there is any.
func (c *endpoints) Update(endpoints *v1.Endpoints) (result *v1.Endpoints, err error) {
	result = &v1.Endpoints{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("endpoints").
		Name(endpoints.Name).
		Body(endpoints).
		Do().
		Into(result)
	return
}

// Delete takes name of the endpoints and deletes it. Returns an error if one occurs.
func (c *endpoints) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("endpoints").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *endpoints) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("endpoints").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the endpoints, and returns the corresponding endpoints object, and an error if there is any.
func (c *endpoints) Get(name string) (result *v1.Endpoints, err error) {
	result = &v1.Endpoints{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("endpoints").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Endpoints that match those selectors.
func (c *endpoints) List(opts api.ListOptions) (result *v1.EndpointsList, err error) {
	result = &v1.EndpointsList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("endpoints").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested endpoints.
func (c *endpoints) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("endpoints").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched endpoints.
func (c *endpoints) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.Endpoints, err error) {
	result = &v1.Endpoints{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("endpoints").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
