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

package unversioned

import (
	api "k8s.io/kubernetes/pkg/api"
	watch "k8s.io/kubernetes/pkg/watch"
)

// ReplicationControllersGetter has a method to return a ReplicationControllerInterface.
// A group's client should implement this interface.
type ReplicationControllersGetter interface {
	ReplicationControllers(namespace string) ReplicationControllerInterface
}

// ReplicationControllerInterface has methods to work with ReplicationController resources.
type ReplicationControllerInterface interface {
	Create(*api.ReplicationController) (*api.ReplicationController, error)
	Update(*api.ReplicationController) (*api.ReplicationController, error)
	UpdateStatus(*api.ReplicationController) (*api.ReplicationController, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*api.ReplicationController, error)
	List(opts api.ListOptions) (*api.ReplicationControllerList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *api.ReplicationController, err error)
	ReplicationControllerExpansion
}

// replicationControllers implements ReplicationControllerInterface
type replicationControllers struct {
	client *CoreClient
	ns     string
}

// newReplicationControllers returns a ReplicationControllers
func newReplicationControllers(c *CoreClient, namespace string) *replicationControllers {
	return &replicationControllers{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a replicationController and creates it.  Returns the server's representation of the replicationController, and an error, if there is any.
func (c *replicationControllers) Create(replicationController *api.ReplicationController) (result *api.ReplicationController, err error) {
	result = &api.ReplicationController{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("replicationcontrollers").
		Body(replicationController).
		Do().
		Into(result)
	return
}

// Update takes the representation of a replicationController and updates it. Returns the server's representation of the replicationController, and an error, if there is any.
func (c *replicationControllers) Update(replicationController *api.ReplicationController) (result *api.ReplicationController, err error) {
	result = &api.ReplicationController{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("replicationcontrollers").
		Name(replicationController.Name).
		Body(replicationController).
		Do().
		Into(result)
	return
}

func (c *replicationControllers) UpdateStatus(replicationController *api.ReplicationController) (result *api.ReplicationController, err error) {
	result = &api.ReplicationController{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("replicationcontrollers").
		Name(replicationController.Name).
		SubResource("status").
		Body(replicationController).
		Do().
		Into(result)
	return
}

// Delete takes name of the replicationController and deletes it. Returns an error if one occurs.
func (c *replicationControllers) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("replicationcontrollers").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *replicationControllers) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("replicationcontrollers").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the replicationController, and returns the corresponding replicationController object, and an error if there is any.
func (c *replicationControllers) Get(name string) (result *api.ReplicationController, err error) {
	result = &api.ReplicationController{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("replicationcontrollers").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ReplicationControllers that match those selectors.
func (c *replicationControllers) List(opts api.ListOptions) (result *api.ReplicationControllerList, err error) {
	result = &api.ReplicationControllerList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("replicationcontrollers").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested replicationControllers.
func (c *replicationControllers) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("replicationcontrollers").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched replicationController.
func (c *replicationControllers) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *api.ReplicationController, err error) {
	result = &api.ReplicationController{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("replicationcontrollers").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
