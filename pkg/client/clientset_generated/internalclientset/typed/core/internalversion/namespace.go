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
	api "k8s.io/kubernetes/pkg/api"
	scheme "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/scheme"
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
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*api.Namespace, error)
	List(opts v1.ListOptions) (*api.NamespaceList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *api.Namespace, err error)
	NamespaceExpansion
}

// namespaces implements NamespaceInterface
type namespaces struct {
	client rest.Interface
}

// newNamespaces returns a Namespaces
func newNamespaces(c *CoreClient) *namespaces {
	return &namespaces{
		client: c.RESTClient(),
	}
}

// Get takes name of the namespace, and returns the corresponding namespace object, and an error if there is any.
func (c *namespaces) Get(name string, options v1.GetOptions) (result *api.Namespace, err error) {
	result = &api.Namespace{}
	err = c.client.Get().
		Resource("namespaces").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Namespaces that match those selectors.
func (c *namespaces) List(opts v1.ListOptions) (result *api.NamespaceList, err error) {
	result = &api.NamespaceList{}
	err = c.client.Get().
		Resource("namespaces").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested namespaces.
func (c *namespaces) Watch(opts v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Resource("namespaces").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
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

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().

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
func (c *namespaces) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Resource("namespaces").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *namespaces) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Resource("namespaces").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Patch applies the patch and returns the patched namespace.
func (c *namespaces) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *api.Namespace, err error) {
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
