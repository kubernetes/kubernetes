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
	watch "k8s.io/apimachinery/pkg/watch"
	api "k8s.io/kubernetes/pkg/api"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
)

// ComponentStatusesGetter has a method to return a ComponentStatusInterface.
// A group's client should implement this interface.
type ComponentStatusesGetter interface {
	ComponentStatuses() ComponentStatusInterface
}

// ComponentStatusInterface has methods to work with ComponentStatus resources.
type ComponentStatusInterface interface {
	Create(*api.ComponentStatus) (*api.ComponentStatus, error)
	Update(*api.ComponentStatus) (*api.ComponentStatus, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string, options v1.GetOptions) (*api.ComponentStatus, error)
	List(opts api.ListOptions) (*api.ComponentStatusList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *api.ComponentStatus, err error)
	ComponentStatusExpansion
}

// componentStatuses implements ComponentStatusInterface
type componentStatuses struct {
	client restclient.Interface
}

// newComponentStatuses returns a ComponentStatuses
func newComponentStatuses(c *CoreClient) *componentStatuses {
	return &componentStatuses{
		client: c.RESTClient(),
	}
}

// Create takes the representation of a componentStatus and creates it.  Returns the server's representation of the componentStatus, and an error, if there is any.
func (c *componentStatuses) Create(componentStatus *api.ComponentStatus) (result *api.ComponentStatus, err error) {
	result = &api.ComponentStatus{}
	err = c.client.Post().
		Resource("componentstatuses").
		Body(componentStatus).
		Do().
		Into(result)
	return
}

// Update takes the representation of a componentStatus and updates it. Returns the server's representation of the componentStatus, and an error, if there is any.
func (c *componentStatuses) Update(componentStatus *api.ComponentStatus) (result *api.ComponentStatus, err error) {
	result = &api.ComponentStatus{}
	err = c.client.Put().
		Resource("componentstatuses").
		Name(componentStatus.Name).
		Body(componentStatus).
		Do().
		Into(result)
	return
}

// Delete takes name of the componentStatus and deletes it. Returns an error if one occurs.
func (c *componentStatuses) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Resource("componentstatuses").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *componentStatuses) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Resource("componentstatuses").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the componentStatus, and returns the corresponding componentStatus object, and an error if there is any.
func (c *componentStatuses) Get(name string, options v1.GetOptions) (result *api.ComponentStatus, err error) {
	result = &api.ComponentStatus{}
	err = c.client.Get().
		Resource("componentstatuses").
		Name(name).
		VersionedParams(&options, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ComponentStatuses that match those selectors.
func (c *componentStatuses) List(opts api.ListOptions) (result *api.ComponentStatusList, err error) {
	result = &api.ComponentStatusList{}
	err = c.client.Get().
		Resource("componentstatuses").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested componentStatuses.
func (c *componentStatuses) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Resource("componentstatuses").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched componentStatus.
func (c *componentStatuses) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *api.ComponentStatus, err error) {
	result = &api.ComponentStatus{}
	err = c.client.Patch(pt).
		Resource("componentstatuses").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
