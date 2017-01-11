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
	rbac "k8s.io/kubernetes/pkg/apis/rbac"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
)

// RolesGetter has a method to return a RoleInterface.
// A group's client should implement this interface.
type RolesGetter interface {
	Roles(namespace string) RoleInterface
}

// RoleInterface has methods to work with Role resources.
type RoleInterface interface {
	Create(*rbac.Role) (*rbac.Role, error)
	Update(*rbac.Role) (*rbac.Role, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string, options v1.GetOptions) (*rbac.Role, error)
	List(opts api.ListOptions) (*rbac.RoleList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *rbac.Role, err error)
	RoleExpansion
}

// roles implements RoleInterface
type roles struct {
	client restclient.Interface
	ns     string
}

// newRoles returns a Roles
func newRoles(c *RbacClient, namespace string) *roles {
	return &roles{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a role and creates it.  Returns the server's representation of the role, and an error, if there is any.
func (c *roles) Create(role *rbac.Role) (result *rbac.Role, err error) {
	result = &rbac.Role{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("roles").
		Body(role).
		Do().
		Into(result)
	return
}

// Update takes the representation of a role and updates it. Returns the server's representation of the role, and an error, if there is any.
func (c *roles) Update(role *rbac.Role) (result *rbac.Role, err error) {
	result = &rbac.Role{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("roles").
		Name(role.Name).
		Body(role).
		Do().
		Into(result)
	return
}

// Delete takes name of the role and deletes it. Returns an error if one occurs.
func (c *roles) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("roles").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *roles) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("roles").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the role, and returns the corresponding role object, and an error if there is any.
func (c *roles) Get(name string, options v1.GetOptions) (result *rbac.Role, err error) {
	result = &rbac.Role{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("roles").
		Name(name).
		VersionedParams(&options, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Roles that match those selectors.
func (c *roles) List(opts api.ListOptions) (result *rbac.RoleList, err error) {
	result = &rbac.RoleList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("roles").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested roles.
func (c *roles) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("roles").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched role.
func (c *roles) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *rbac.Role, err error) {
	result = &rbac.Role{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("roles").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
