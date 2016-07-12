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

// ClusterProtectedAttributesGetter has a method to return a ClusterProtectedAttributeInterface.
// A group's client should implement this interface.
type ClusterProtectedAttributesGetter interface {
	ClusterProtectedAttributes() ClusterProtectedAttributeInterface
}

// ClusterProtectedAttributeInterface has methods to work with ClusterProtectedAttribute resources.
type ClusterProtectedAttributeInterface interface {
	Create(*rbac.ClusterProtectedAttribute) (*rbac.ClusterProtectedAttribute, error)
	Update(*rbac.ClusterProtectedAttribute) (*rbac.ClusterProtectedAttribute, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*rbac.ClusterProtectedAttribute, error)
	List(opts api.ListOptions) (*rbac.ClusterProtectedAttributeList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte) (result *rbac.ClusterProtectedAttribute, err error)
	ClusterProtectedAttributeExpansion
}

// clusterProtectedAttributes implements ClusterProtectedAttributeInterface
type clusterProtectedAttributes struct {
	client *RbacClient
}

// newClusterProtectedAttributes returns a ClusterProtectedAttributes
func newClusterProtectedAttributes(c *RbacClient) *clusterProtectedAttributes {
	return &clusterProtectedAttributes{
		client: c,
	}
}

// Create takes the representation of a clusterProtectedAttribute and creates it.  Returns the server's representation of the clusterProtectedAttribute, and an error, if there is any.
func (c *clusterProtectedAttributes) Create(clusterProtectedAttribute *rbac.ClusterProtectedAttribute) (result *rbac.ClusterProtectedAttribute, err error) {
	result = &rbac.ClusterProtectedAttribute{}
	err = c.client.Post().
		Resource("clusterprotectedattributes").
		Body(clusterProtectedAttribute).
		Do().
		Into(result)
	return
}

// Update takes the representation of a clusterProtectedAttribute and updates it. Returns the server's representation of the clusterProtectedAttribute, and an error, if there is any.
func (c *clusterProtectedAttributes) Update(clusterProtectedAttribute *rbac.ClusterProtectedAttribute) (result *rbac.ClusterProtectedAttribute, err error) {
	result = &rbac.ClusterProtectedAttribute{}
	err = c.client.Put().
		Resource("clusterprotectedattributes").
		Name(clusterProtectedAttribute.Name).
		Body(clusterProtectedAttribute).
		Do().
		Into(result)
	return
}

// Delete takes name of the clusterProtectedAttribute and deletes it. Returns an error if one occurs.
func (c *clusterProtectedAttributes) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Resource("clusterprotectedattributes").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *clusterProtectedAttributes) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Resource("clusterprotectedattributes").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the clusterProtectedAttribute, and returns the corresponding clusterProtectedAttribute object, and an error if there is any.
func (c *clusterProtectedAttributes) Get(name string) (result *rbac.ClusterProtectedAttribute, err error) {
	result = &rbac.ClusterProtectedAttribute{}
	err = c.client.Get().
		Resource("clusterprotectedattributes").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ClusterProtectedAttributes that match those selectors.
func (c *clusterProtectedAttributes) List(opts api.ListOptions) (result *rbac.ClusterProtectedAttributeList, err error) {
	result = &rbac.ClusterProtectedAttributeList{}
	err = c.client.Get().
		Resource("clusterprotectedattributes").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested clusterProtectedAttributes.
func (c *clusterProtectedAttributes) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Resource("clusterprotectedattributes").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched clusterProtectedAttribute.
func (c *clusterProtectedAttributes) Patch(name string, pt api.PatchType, data []byte) (result *rbac.ClusterProtectedAttribute, err error) {
	result = &rbac.ClusterProtectedAttribute{}
	err = c.client.Patch(pt).
		Resource("clusterprotectedattributes").
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
