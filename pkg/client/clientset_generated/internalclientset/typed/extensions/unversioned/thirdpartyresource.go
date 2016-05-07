/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	api "k8s.io/kubernetes/pkg/api"
	extensions "k8s.io/kubernetes/pkg/apis/extensions"
	watch "k8s.io/kubernetes/pkg/watch"
)

// ThirdPartyResourcesGetter has a method to return a ThirdPartyResourceInterface.
// A group's client should implement this interface.
type ThirdPartyResourcesGetter interface {
	ThirdPartyResources() ThirdPartyResourceInterface
}

// ThirdPartyResourceInterface has methods to work with ThirdPartyResource resources.
type ThirdPartyResourceInterface interface {
	Create(*extensions.ThirdPartyResource) (*extensions.ThirdPartyResource, error)
	Update(*extensions.ThirdPartyResource) (*extensions.ThirdPartyResource, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*extensions.ThirdPartyResource, error)
	List(opts api.ListOptions) (*extensions.ThirdPartyResourceList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
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
func (c *thirdPartyResources) Create(thirdPartyResource *extensions.ThirdPartyResource) (result *extensions.ThirdPartyResource, err error) {
	result = &extensions.ThirdPartyResource{}
	err = c.client.Post().
		Resource("thirdpartyresources").
		Body(thirdPartyResource).
		Do().
		Into(result)
	return
}

// Update takes the representation of a thirdPartyResource and updates it. Returns the server's representation of the thirdPartyResource, and an error, if there is any.
func (c *thirdPartyResources) Update(thirdPartyResource *extensions.ThirdPartyResource) (result *extensions.ThirdPartyResource, err error) {
	result = &extensions.ThirdPartyResource{}
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
func (c *thirdPartyResources) Get(name string) (result *extensions.ThirdPartyResource, err error) {
	result = &extensions.ThirdPartyResource{}
	err = c.client.Get().
		Resource("thirdpartyresources").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ThirdPartyResources that match those selectors.
func (c *thirdPartyResources) List(opts api.ListOptions) (result *extensions.ThirdPartyResourceList, err error) {
	result = &extensions.ThirdPartyResourceList{}
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
