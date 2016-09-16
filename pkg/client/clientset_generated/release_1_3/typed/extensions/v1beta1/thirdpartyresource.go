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

package v1beta1

import (
	api "k8s.io/kubernetes/pkg/api"
	v1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	watch "k8s.io/kubernetes/pkg/watch"
)

// ThirdPartyResourcesGetter has a method to return a ThirdPartyResourceInterface.
// A group's client should implement this interface.
type ThirdPartyResourcesGetter interface {
	ThirdPartyResources() ThirdPartyResourceInterface
}

// ThirdPartyResourceInterface has methods to work with ThirdPartyResource resources.
type ThirdPartyResourceInterface interface {
	Create(*v1beta1.ThirdPartyResource) (*v1beta1.ThirdPartyResource, error)
	Update(*v1beta1.ThirdPartyResource) (*v1beta1.ThirdPartyResource, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*v1beta1.ThirdPartyResource, error)
	List(opts api.ListOptions) (*v1beta1.ThirdPartyResourceList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte) (result *v1beta1.ThirdPartyResource, err error)
	ThirdPartyResourceExpansion
}

// thirdPartyResources implements ThirdPartyResourceInterface
type thirdPartyResources struct {
	client *ExtensionsClient
}

// newThirdPartyResources returns a ThirdPartyResources
func newThirdPartyResources(c *ExtensionsClient) *thirdPartyResources {
	return &thirdPartyResources{
		client: c,
	}
}

// Create takes the representation of a thirdPartyResource and creates it.  Returns the server's representation of the thirdPartyResource, and an error, if there is any.
func (c *thirdPartyResources) Create(thirdPartyResource *v1beta1.ThirdPartyResource) (result *v1beta1.ThirdPartyResource, err error) {
	result = &v1beta1.ThirdPartyResource{}
	err = c.client.Post().
		Resource("thirdpartyresources").
		Body(thirdPartyResource).
		Do().
		Into(result)
	return
}

// Update takes the representation of a thirdPartyResource and updates it. Returns the server's representation of the thirdPartyResource, and an error, if there is any.
func (c *thirdPartyResources) Update(thirdPartyResource *v1beta1.ThirdPartyResource) (result *v1beta1.ThirdPartyResource, err error) {
	result = &v1beta1.ThirdPartyResource{}
	err = c.client.Put().
		Resource("thirdpartyresources").
		Name(thirdPartyResource.Name).
		Body(thirdPartyResource).
		Do().
		Into(result)
	return
}

// Delete takes name of the thirdPartyResource and deletes it. Returns an error if one occurs.
func (c *thirdPartyResources) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Resource("thirdpartyresources").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *thirdPartyResources) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Resource("thirdpartyresources").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the thirdPartyResource, and returns the corresponding thirdPartyResource object, and an error if there is any.
func (c *thirdPartyResources) Get(name string) (result *v1beta1.ThirdPartyResource, err error) {
	result = &v1beta1.ThirdPartyResource{}
	err = c.client.Get().
		Resource("thirdpartyresources").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ThirdPartyResources that match those selectors.
func (c *thirdPartyResources) List(opts api.ListOptions) (result *v1beta1.ThirdPartyResourceList, err error) {
	result = &v1beta1.ThirdPartyResourceList{}
	err = c.client.Get().
		Resource("thirdpartyresources").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested thirdPartyResources.
func (c *thirdPartyResources) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Resource("thirdpartyresources").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched thirdPartyResource.
func (c *thirdPartyResources) Patch(name string, pt api.PatchType, data []byte) (result *v1beta1.ThirdPartyResource, err error) {
	result = &v1beta1.ThirdPartyResource{}
	err = c.client.Patch(pt).
		Resource("thirdpartyresources").
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
