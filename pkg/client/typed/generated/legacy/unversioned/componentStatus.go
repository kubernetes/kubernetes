/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	watch "k8s.io/kubernetes/pkg/watch"
)

// ComponentStatusNamespacer has methods to work with ComponentStatus resources in a namespace
type ComponentStatusNamespacer interface {
	ComponentStatus(namespace string) ComponentStatusInterface
}

// ComponentStatusInterface has methods to work with ComponentStatus resources.
type ComponentStatusInterface interface {
	Create(*api.ComponentStatus) (*api.ComponentStatus, error)
	Update(*api.ComponentStatus) (*api.ComponentStatus, error)
	Delete(name string, options *api.DeleteOptions) error
	Get(name string) (*api.ComponentStatus, error)
	List(opts unversioned.ListOptions) (*api.ComponentStatusList, error)
	Watch(opts unversioned.ListOptions) (watch.Interface, error)
}

// componentStatus implements ComponentStatusInterface
type componentStatus struct {
	client *LegacyClient
	ns     string
}

// newComponentStatus returns a ComponentStatus
func newComponentStatus(c *LegacyClient, namespace string) *componentStatus {
	return &componentStatus{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a componentStatus and creates it.  Returns the server's representation of the componentStatus, and an error, if there is any.
func (c *componentStatus) Create(componentStatus *api.ComponentStatus) (result *api.ComponentStatus, err error) {
	result = &api.ComponentStatus{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("componentStatus").
		Body(componentStatus).
		Do().
		Into(result)
	return
}

// Update takes the representation of a componentStatus and updates it. Returns the server's representation of the componentStatus, and an error, if there is any.
func (c *componentStatus) Update(componentStatus *api.ComponentStatus) (result *api.ComponentStatus, err error) {
	result = &api.ComponentStatus{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("componentStatus").
		Name(componentStatus.Name).
		Body(componentStatus).
		Do().
		Into(result)
	return
}

// Delete takes name of the componentStatus and deletes it. Returns an error if one occurs.
func (c *componentStatus) Delete(name string, options *api.DeleteOptions) error {
	if options == nil {
		return c.client.Delete().Namespace(c.ns).Resource("componentStatus").Name(name).Do().Error()
	}
	body, err := api.Scheme.EncodeToVersion(options, c.client.APIVersion().String())
	if err != nil {
		return err
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("componentStatus").
		Name(name).
		Body(body).
		Do().
		Error()
}

// Get takes name of the componentStatus, and returns the corresponding componentStatus object, and an error if there is any.
func (c *componentStatus) Get(name string) (result *api.ComponentStatus, err error) {
	result = &api.ComponentStatus{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("componentStatus").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ComponentStatus that match those selectors.
func (c *componentStatus) List(opts unversioned.ListOptions) (result *api.ComponentStatusList, err error) {
	result = &api.ComponentStatusList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("componentStatus").
		VersionedParams(&opts, api.Scheme).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested componentStatus.
func (c *componentStatus) Watch(opts unversioned.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("componentStatus").
		VersionedParams(&opts, api.Scheme).
		Watch()
}
