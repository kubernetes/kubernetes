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

package internalversion

import (
	apifederation "k8s.io/kubernetes/apifederator/pkg/apis/apifederation"
	api "k8s.io/kubernetes/pkg/api"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	watch "k8s.io/kubernetes/pkg/watch"
)

// APIServersGetter has a method to return a APIServerInterface.
// A group's client should implement this interface.
type APIServersGetter interface {
	APIServers() APIServerInterface
}

// APIServerInterface has methods to work with APIServer resources.
type APIServerInterface interface {
	Create(*apifederation.APIServer) (*apifederation.APIServer, error)
	Update(*apifederation.APIServer) (*apifederation.APIServer, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*apifederation.APIServer, error)
	List(opts api.ListOptions) (*apifederation.APIServerList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *apifederation.APIServer, err error)
	APIServerExpansion
}

// aPIServers implements APIServerInterface
type aPIServers struct {
	client restclient.Interface
}

// newAPIServers returns a APIServers
func newAPIServers(c *ApifederationClient) *aPIServers {
	return &aPIServers{
		client: c.RESTClient(),
	}
}

// Create takes the representation of a aPIServer and creates it.  Returns the server's representation of the aPIServer, and an error, if there is any.
func (c *aPIServers) Create(aPIServer *apifederation.APIServer) (result *apifederation.APIServer, err error) {
	result = &apifederation.APIServer{}
	err = c.client.Post().
		Resource("apiservers").
		Body(aPIServer).
		Do().
		Into(result)
	return
}

// Update takes the representation of a aPIServer and updates it. Returns the server's representation of the aPIServer, and an error, if there is any.
func (c *aPIServers) Update(aPIServer *apifederation.APIServer) (result *apifederation.APIServer, err error) {
	result = &apifederation.APIServer{}
	err = c.client.Put().
		Resource("apiservers").
		Name(aPIServer.Name).
		Body(aPIServer).
		Do().
		Into(result)
	return
}

// Delete takes name of the aPIServer and deletes it. Returns an error if one occurs.
func (c *aPIServers) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Resource("apiservers").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *aPIServers) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Resource("apiservers").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the aPIServer, and returns the corresponding aPIServer object, and an error if there is any.
func (c *aPIServers) Get(name string) (result *apifederation.APIServer, err error) {
	result = &apifederation.APIServer{}
	err = c.client.Get().
		Resource("apiservers").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of APIServers that match those selectors.
func (c *aPIServers) List(opts api.ListOptions) (result *apifederation.APIServerList, err error) {
	result = &apifederation.APIServerList{}
	err = c.client.Get().
		Resource("apiservers").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested aPIServers.
func (c *aPIServers) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Resource("apiservers").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched aPIServer.
func (c *aPIServers) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *apifederation.APIServer, err error) {
	result = &apifederation.APIServer{}
	err = c.client.Patch(pt).
		Resource("apiservers").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
