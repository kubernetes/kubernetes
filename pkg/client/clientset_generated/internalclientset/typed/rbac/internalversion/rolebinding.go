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

package internalversion

import (
	api "k8s.io/kubernetes/pkg/api"
	rbac "k8s.io/kubernetes/pkg/apis/rbac"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	watch "k8s.io/kubernetes/pkg/watch"
)

// RoleBindingsGetter has a method to return a RoleBindingInterface.
// A group's client should implement this interface.
type RoleBindingsGetter interface {
	RoleBindings(namespace string) RoleBindingInterface
}

// RoleBindingInterface has methods to work with RoleBinding resources.
type RoleBindingInterface interface {
	Create(*rbac.RoleBinding) (*rbac.RoleBinding, error)
	Update(*rbac.RoleBinding) (*rbac.RoleBinding, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*rbac.RoleBinding, error)
	List(opts api.ListOptions) (*rbac.RoleBindingList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *rbac.RoleBinding, err error)
	RoleBindingExpansion
}

// roleBindings implements RoleBindingInterface
type roleBindings struct {
	client restclient.Interface
	ns     string
}

// newRoleBindings returns a RoleBindings
func newRoleBindings(c *RbacClient, namespace string) *roleBindings {
	return &roleBindings{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a roleBinding and creates it.  Returns the server's representation of the roleBinding, and an error, if there is any.
func (c *roleBindings) Create(roleBinding *rbac.RoleBinding) (result *rbac.RoleBinding, err error) {
	result = &rbac.RoleBinding{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("rolebindings").
		Body(roleBinding).
		Do().
		Into(result)
	return
}

// Update takes the representation of a roleBinding and updates it. Returns the server's representation of the roleBinding, and an error, if there is any.
func (c *roleBindings) Update(roleBinding *rbac.RoleBinding) (result *rbac.RoleBinding, err error) {
	result = &rbac.RoleBinding{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("rolebindings").
		Name(roleBinding.Name).
		Body(roleBinding).
		Do().
		Into(result)
	return
}

// Delete takes name of the roleBinding and deletes it. Returns an error if one occurs.
func (c *roleBindings) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("rolebindings").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *roleBindings) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("rolebindings").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the roleBinding, and returns the corresponding roleBinding object, and an error if there is any.
func (c *roleBindings) Get(name string) (result *rbac.RoleBinding, err error) {
	result = &rbac.RoleBinding{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("rolebindings").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of RoleBindings that match those selectors.
func (c *roleBindings) List(opts api.ListOptions) (result *rbac.RoleBindingList, err error) {
	result = &rbac.RoleBindingList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("rolebindings").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested roleBindings.
func (c *roleBindings) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("rolebindings").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched roleBinding.
func (c *roleBindings) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *rbac.RoleBinding, err error) {
	result = &rbac.RoleBinding{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("rolebindings").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
