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
	rbac "k8s.io/kubernetes/pkg/apis/rbac"
	watch "k8s.io/kubernetes/pkg/watch"
)

// ProtectedAttributesGetter has a method to return a ProtectedAttributeInterface.
// A group's client should implement this interface.
type ProtectedAttributesGetter interface {
	ProtectedAttributes(namespace string) ProtectedAttributeInterface
}

// ProtectedAttributeInterface has methods to work with ProtectedAttribute resources.
type ProtectedAttributeInterface interface {
	Create(*rbac.ProtectedAttribute) (*rbac.ProtectedAttribute, error)
	Update(*rbac.ProtectedAttribute) (*rbac.ProtectedAttribute, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*rbac.ProtectedAttribute, error)
	List(opts api.ListOptions) (*rbac.ProtectedAttributeList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte) (result *rbac.ProtectedAttribute, err error)
	ProtectedAttributeExpansion
}

// protectedAttributes implements ProtectedAttributeInterface
type protectedAttributes struct {
	client *RbacClient
	ns     string
}

// newProtectedAttributes returns a ProtectedAttributes
func newProtectedAttributes(c *RbacClient, namespace string) *protectedAttributes {
	return &protectedAttributes{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a protectedAttribute and creates it.  Returns the server's representation of the protectedAttribute, and an error, if there is any.
func (c *protectedAttributes) Create(protectedAttribute *rbac.ProtectedAttribute) (result *rbac.ProtectedAttribute, err error) {
	result = &rbac.ProtectedAttribute{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("protectedattributes").
		Body(protectedAttribute).
		Do().
		Into(result)
	return
}

// Update takes the representation of a protectedAttribute and updates it. Returns the server's representation of the protectedAttribute, and an error, if there is any.
func (c *protectedAttributes) Update(protectedAttribute *rbac.ProtectedAttribute) (result *rbac.ProtectedAttribute, err error) {
	result = &rbac.ProtectedAttribute{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("protectedattributes").
		Name(protectedAttribute.Name).
		Body(protectedAttribute).
		Do().
		Into(result)
	return
}

// Delete takes name of the protectedAttribute and deletes it. Returns an error if one occurs.
func (c *protectedAttributes) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("protectedattributes").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *protectedAttributes) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("protectedattributes").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the protectedAttribute, and returns the corresponding protectedAttribute object, and an error if there is any.
func (c *protectedAttributes) Get(name string) (result *rbac.ProtectedAttribute, err error) {
	result = &rbac.ProtectedAttribute{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("protectedattributes").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ProtectedAttributes that match those selectors.
func (c *protectedAttributes) List(opts api.ListOptions) (result *rbac.ProtectedAttributeList, err error) {
	result = &rbac.ProtectedAttributeList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("protectedattributes").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested protectedAttributes.
func (c *protectedAttributes) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("protectedattributes").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched protectedAttribute.
func (c *protectedAttributes) Patch(name string, pt api.PatchType, data []byte) (result *rbac.ProtectedAttribute, err error) {
	result = &rbac.ProtectedAttribute{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("protectedattributes").
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
