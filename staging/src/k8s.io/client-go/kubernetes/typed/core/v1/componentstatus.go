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
	api "k8s.io/client-go/pkg/api"
	v1 "k8s.io/client-go/pkg/api/v1"
	meta_v1 "k8s.io/client-go/pkg/apis/meta/v1"
	watch "k8s.io/client-go/pkg/watch"
	rest "k8s.io/client-go/rest"
)

// ComponentStatusesGetter has a method to return a ComponentStatusInterface.
// A group's client should implement this interface.
type ComponentStatusesGetter interface {
	ComponentStatuses() ComponentStatusInterface
}

// ComponentStatusInterface has methods to work with ComponentStatus resources.
type ComponentStatusInterface interface {
	Create(*v1.ComponentStatus) (*v1.ComponentStatus, error)
	Update(*v1.ComponentStatus) (*v1.ComponentStatus, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options meta_v1.GetOptions) (*v1.ComponentStatus, error)
	List(opts v1.ListOptions) (*v1.ComponentStatusList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.ComponentStatus, err error)
	ComponentStatusExpansion
}

// componentStatuses implements ComponentStatusInterface
type componentStatuses struct {
	client rest.Interface
}

// newComponentStatuses returns a ComponentStatuses
func newComponentStatuses(c *CoreV1Client) *componentStatuses {
	return &componentStatuses{
		client: c.RESTClient(),
	}
}

// Create takes the representation of a componentStatus and creates it.  Returns the server's representation of the componentStatus, and an error, if there is any.
func (c *componentStatuses) Create(componentStatus *v1.ComponentStatus) (result *v1.ComponentStatus, err error) {
	result = &v1.ComponentStatus{}
	err = c.client.Post().
		Resource("componentstatuses").
		Body(componentStatus).
		Do().
		Into(result)
	return
}

// Update takes the representation of a componentStatus and updates it. Returns the server's representation of the componentStatus, and an error, if there is any.
func (c *componentStatuses) Update(componentStatus *v1.ComponentStatus) (result *v1.ComponentStatus, err error) {
	result = &v1.ComponentStatus{}
	err = c.client.Put().
		Resource("componentstatuses").
		Name(componentStatus.Name).
		Body(componentStatus).
		Do().
		Into(result)
	return
}

// Delete takes name of the componentStatus and deletes it. Returns an error if one occurs.
func (c *componentStatuses) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Resource("componentstatuses").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *componentStatuses) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Resource("componentstatuses").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the componentStatus, and returns the corresponding componentStatus object, and an error if there is any.
func (c *componentStatuses) Get(name string, options meta_v1.GetOptions) (result *v1.ComponentStatus, err error) {
	result = &v1.ComponentStatus{}
	err = c.client.Get().
		Resource("componentstatuses").
		Name(name).
		VersionedParams(&options, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ComponentStatuses that match those selectors.
func (c *componentStatuses) List(opts v1.ListOptions) (result *v1.ComponentStatusList, err error) {
	result = &v1.ComponentStatusList{}
	err = c.client.Get().
		Resource("componentstatuses").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested componentStatuses.
func (c *componentStatuses) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Resource("componentstatuses").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched componentStatus.
func (c *componentStatuses) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.ComponentStatus, err error) {
	result = &v1.ComponentStatus{}
	err = c.client.Patch(pt).
		Resource("componentstatuses").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
