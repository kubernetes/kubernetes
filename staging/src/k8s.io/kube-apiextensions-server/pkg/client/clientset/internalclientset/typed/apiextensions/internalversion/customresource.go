/*
Copyright 2017 The Kubernetes Authors.

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
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	rest "k8s.io/client-go/rest"
	apiextensions "k8s.io/kube-apiextensions-server/pkg/apis/apiextensions"
	scheme "k8s.io/kube-apiextensions-server/pkg/client/clientset/internalclientset/scheme"
)

// CustomResourcesGetter has a method to return a CustomResourceInterface.
// A group's client should implement this interface.
type CustomResourcesGetter interface {
	CustomResources() CustomResourceInterface
}

// CustomResourceInterface has methods to work with CustomResource resources.
type CustomResourceInterface interface {
	Create(*apiextensions.CustomResource) (*apiextensions.CustomResource, error)
	Update(*apiextensions.CustomResource) (*apiextensions.CustomResource, error)
	UpdateStatus(*apiextensions.CustomResource) (*apiextensions.CustomResource, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*apiextensions.CustomResource, error)
	List(opts v1.ListOptions) (*apiextensions.CustomResourceList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *apiextensions.CustomResource, err error)
	CustomResourceExpansion
}

// customResources implements CustomResourceInterface
type customResources struct {
	client rest.Interface
}

// newCustomResources returns a CustomResources
func newCustomResources(c *ApiextensionsClient) *customResources {
	return &customResources{
		client: c.RESTClient(),
	}
}

// Create takes the representation of a customResource and creates it.  Returns the server's representation of the customResource, and an error, if there is any.
func (c *customResources) Create(customResource *apiextensions.CustomResource) (result *apiextensions.CustomResource, err error) {
	result = &apiextensions.CustomResource{}
	err = c.client.Post().
		Resource("customresources").
		Body(customResource).
		Do().
		Into(result)
	return
}

// Update takes the representation of a customResource and updates it. Returns the server's representation of the customResource, and an error, if there is any.
func (c *customResources) Update(customResource *apiextensions.CustomResource) (result *apiextensions.CustomResource, err error) {
	result = &apiextensions.CustomResource{}
	err = c.client.Put().
		Resource("customresources").
		Name(customResource.Name).
		Body(customResource).
		Do().
		Into(result)
	return
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclientstatus=false comment above the type to avoid generating UpdateStatus().

func (c *customResources) UpdateStatus(customResource *apiextensions.CustomResource) (result *apiextensions.CustomResource, err error) {
	result = &apiextensions.CustomResource{}
	err = c.client.Put().
		Resource("customresources").
		Name(customResource.Name).
		SubResource("status").
		Body(customResource).
		Do().
		Into(result)
	return
}

// Delete takes name of the customResource and deletes it. Returns an error if one occurs.
func (c *customResources) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Resource("customresources").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *customResources) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Resource("customresources").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the customResource, and returns the corresponding customResource object, and an error if there is any.
func (c *customResources) Get(name string, options v1.GetOptions) (result *apiextensions.CustomResource, err error) {
	result = &apiextensions.CustomResource{}
	err = c.client.Get().
		Resource("customresources").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of CustomResources that match those selectors.
func (c *customResources) List(opts v1.ListOptions) (result *apiextensions.CustomResourceList, err error) {
	result = &apiextensions.CustomResourceList{}
	err = c.client.Get().
		Resource("customresources").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested customResources.
func (c *customResources) Watch(opts v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Resource("customresources").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched customResource.
func (c *customResources) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *apiextensions.CustomResource, err error) {
	result = &apiextensions.CustomResource{}
	err = c.client.Patch(pt).
		Resource("customresources").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
