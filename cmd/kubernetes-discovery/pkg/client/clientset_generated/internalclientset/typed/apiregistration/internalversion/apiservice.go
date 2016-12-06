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
	apiregistration "k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/apis/apiregistration"
	api "k8s.io/kubernetes/pkg/api"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	watch "k8s.io/kubernetes/pkg/watch"
)

// APIServicesGetter has a method to return a APIServiceInterface.
// A group's client should implement this interface.
type APIServicesGetter interface {
	APIServices() APIServiceInterface
}

// APIServiceInterface has methods to work with APIService resources.
type APIServiceInterface interface {
	Create(*apiregistration.APIService) (*apiregistration.APIService, error)
	Update(*apiregistration.APIService) (*apiregistration.APIService, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*apiregistration.APIService, error)
	List(opts api.ListOptions) (*apiregistration.APIServiceList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *apiregistration.APIService, err error)
	APIServiceExpansion
}

// aPIServices implements APIServiceInterface
type aPIServices struct {
	client restclient.Interface
}

// newAPIServices returns a APIServices
func newAPIServices(c *ApiregistrationClient) *aPIServices {
	return &aPIServices{
		client: c.RESTClient(),
	}
}

// Create takes the representation of a aPIService and creates it.  Returns the server's representation of the aPIService, and an error, if there is any.
func (c *aPIServices) Create(aPIService *apiregistration.APIService) (result *apiregistration.APIService, err error) {
	result = &apiregistration.APIService{}
	err = c.client.Post().
		Resource("apiservices").
		Body(aPIService).
		Do().
		Into(result)
	return
}

// Update takes the representation of a aPIService and updates it. Returns the server's representation of the aPIService, and an error, if there is any.
func (c *aPIServices) Update(aPIService *apiregistration.APIService) (result *apiregistration.APIService, err error) {
	result = &apiregistration.APIService{}
	err = c.client.Put().
		Resource("apiservices").
		Name(aPIService.Name).
		Body(aPIService).
		Do().
		Into(result)
	return
}

// Delete takes name of the aPIService and deletes it. Returns an error if one occurs.
func (c *aPIServices) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Resource("apiservices").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *aPIServices) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Resource("apiservices").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the aPIService, and returns the corresponding aPIService object, and an error if there is any.
func (c *aPIServices) Get(name string) (result *apiregistration.APIService, err error) {
	result = &apiregistration.APIService{}
	err = c.client.Get().
		Resource("apiservices").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of APIServices that match those selectors.
func (c *aPIServices) List(opts api.ListOptions) (result *apiregistration.APIServiceList, err error) {
	result = &apiregistration.APIServiceList{}
	err = c.client.Get().
		Resource("apiservices").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested aPIServices.
func (c *aPIServices) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Resource("apiservices").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched aPIService.
func (c *aPIServices) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *apiregistration.APIService, err error) {
	result = &apiregistration.APIService{}
	err = c.client.Patch(pt).
		Resource("apiservices").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
