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

// ThirdPartyResourceDatasGetter has a method to return a ThirdPartyResourceDataInterface.
// A group's client should implement this interface.
type ThirdPartyResourceDatasGetter interface {
	ThirdPartyResourceDatas(namespace string) ThirdPartyResourceDataInterface
}

// ThirdPartyResourceDataInterface has methods to work with ThirdPartyResourceData resources.
type ThirdPartyResourceDataInterface interface {
	Create(*extensions.ThirdPartyResourceData) (*extensions.ThirdPartyResourceData, error)
	Update(*extensions.ThirdPartyResourceData) (*extensions.ThirdPartyResourceData, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*extensions.ThirdPartyResourceData, error)
	List(opts api.ListOptions) (*extensions.ThirdPartyResourceDataList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	ThirdPartyResourceDataExpansion
}

// thirdPartyResourceDatas implements ThirdPartyResourceDataInterface
type thirdPartyResourceDatas struct {
	client *ExtensionsClient
	ns     string
}

// newThirdPartyResourceDatas returns a ThirdPartyResourceDatas
func newThirdPartyResourceDatas(c *ExtensionsClient, namespace string) *thirdPartyResourceDatas {
	return &thirdPartyResourceDatas{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a thirdPartyResourceData and creates it.  Returns the server's representation of the thirdPartyResourceData, and an error, if there is any.
func (c *thirdPartyResourceDatas) Create(thirdPartyResourceData *extensions.ThirdPartyResourceData) (result *extensions.ThirdPartyResourceData, err error) {
	result = &extensions.ThirdPartyResourceData{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("thirdpartyresourcedatas").
		Body(thirdPartyResourceData).
		Do().
		Into(result)
	return
}

// Update takes the representation of a thirdPartyResourceData and updates it. Returns the server's representation of the thirdPartyResourceData, and an error, if there is any.
func (c *thirdPartyResourceDatas) Update(thirdPartyResourceData *extensions.ThirdPartyResourceData) (result *extensions.ThirdPartyResourceData, err error) {
	result = &extensions.ThirdPartyResourceData{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("thirdpartyresourcedatas").
		Name(thirdPartyResourceData.Name).
		Body(thirdPartyResourceData).
		Do().
		Into(result)
	return
}

// Delete takes name of the thirdPartyResourceData and deletes it. Returns an error if one occurs.
func (c *thirdPartyResourceDatas) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("thirdpartyresourcedatas").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *thirdPartyResourceDatas) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("thirdpartyresourcedatas").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the thirdPartyResourceData, and returns the corresponding thirdPartyResourceData object, and an error if there is any.
func (c *thirdPartyResourceDatas) Get(name string) (result *extensions.ThirdPartyResourceData, err error) {
	result = &extensions.ThirdPartyResourceData{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("thirdpartyresourcedatas").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ThirdPartyResourceDatas that match those selectors.
func (c *thirdPartyResourceDatas) List(opts api.ListOptions) (result *extensions.ThirdPartyResourceDataList, err error) {
	result = &extensions.ThirdPartyResourceDataList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("thirdpartyresourcedatas").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested thirdPartyResourceDatas.
func (c *thirdPartyResourceDatas) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("thirdpartyresourcedatas").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}
