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

// NamespacesGetter has a method to return a NamespaceInterface.
// A group's client should implement this interface.
type NamespacesGetter interface {
	Namespaces() NamespaceInterface
}

// NamespaceInterface has methods to work with Namespace resources.
type NamespaceInterface interface {
	Create(*api.Namespace) (*api.Namespace, error)
	Update(*api.Namespace) (*api.Namespace, error)
	UpdateStatus(*api.Namespace) (*api.Namespace, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*api.Namespace, error)
	List(opts api.ListOptions) (*api.NamespaceList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *api.Namespace, err error)
	NamespaceExpansion
}

// namespaces implements NamespaceInterface
type namespaces struct {
	client *CoreClient
}

// newNamespaces returns a Namespaces
func newNamespaces(c *CoreClient) *namespaces {
	return &namespaces{
		client: c,
	}
}

// Create takes the representation of a namespace and creates it.  Returns the server's representation of the namespace, and an error, if there is any.
func (c *namespaces) Create(namespace *api.Namespace) (result *api.Namespace, err error) {
	result = &api.Namespace{}
	err = c.client.Post().
		Resource("namespaces").
		Body(namespace).
		Do().
		Into(result)
	return
}

// Update takes the representation of a namespace and updates it. Returns the server's representation of the namespace, and an error, if there is any.
func (c *namespaces) Update(namespace *api.Namespace) (result *api.Namespace, err error) {
	result = &api.Namespace{}
	err = c.client.Put().
		Resource("namespaces").
		Name(namespace.Name).
		Body(namespace).
		Do().
		Into(result)
	return
}

func (c *namespaces) UpdateStatus(namespace *api.Namespace) (result *api.Namespace, err error) {
	result = &api.Namespace{}
	err = c.client.Put().
		Resource("namespaces").
		Name(namespace.Name).
		SubResource("status").
		Body(namespace).
		Do().
		Into(result)
	return
}

// Delete takes name of the namespace and deletes it. Returns an error if one occurs.
func (c *namespaces) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Resource("namespaces").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *namespaces) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Resource("namespaces").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the namespace, and returns the corresponding namespace object, and an error if there is any.
func (c *namespaces) Get(name string) (result *api.Namespace, err error) {
	result = &api.Namespace{}
	err = c.client.Get().
		Resource("namespaces").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Namespaces that match those selectors.
func (c *namespaces) List(opts api.ListOptions) (result *api.NamespaceList, err error) {
	result = &api.NamespaceList{}
	err = c.client.Get().
		Resource("namespaces").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested namespaces.
func (c *namespaces) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Resource("namespaces").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched namespace.
func (c *namespaces) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *api.Namespace, err error) {
	result = &api.Namespace{}
	err = c.client.Patch(pt).
		Resource("namespaces").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
