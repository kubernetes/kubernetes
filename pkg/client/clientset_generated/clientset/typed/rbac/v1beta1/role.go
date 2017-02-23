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

package v1beta1

import (
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	rest "k8s.io/client-go/rest"
	v1beta1 "k8s.io/kubernetes/pkg/apis/rbac/v1beta1"
	scheme "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/scheme"
)

// RolesGetter has a method to return a RoleInterface.
// A group's client should implement this interface.
type RolesGetter interface {
	Roles(namespace string) RoleInterface
}

// RoleInterface has methods to work with Role resources.
type RoleInterface interface {
	Create(*v1beta1.Role) (*v1beta1.Role, error)
	Update(*v1beta1.Role) (*v1beta1.Role, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*v1beta1.Role, error)
	List(opts v1.ListOptions) (*v1beta1.RoleList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1beta1.Role, err error)
	RoleExpansion
}

// roles implements RoleInterface
type roles struct {
	client rest.Interface
	ns     string
}

// newRoles returns a Roles
func newRoles(c *RbacV1beta1Client, namespace string) *roles {
	return &roles{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a role and creates it.  Returns the server's representation of the role, and an error, if there is any.
func (c *roles) Create(role *v1beta1.Role) (result *v1beta1.Role, err error) {
	result = &v1beta1.Role{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("roles").
		Body(role).
		Do().
		Into(result)
	return
}

// Update takes the representation of a role and updates it. Returns the server's representation of the role, and an error, if there is any.
func (c *roles) Update(role *v1beta1.Role) (result *v1beta1.Role, err error) {
	result = &v1beta1.Role{}
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
func (c *roles) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("roles").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *roles) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("roles").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the role, and returns the corresponding role object, and an error if there is any.
func (c *roles) Get(name string, options v1.GetOptions) (result *v1beta1.Role, err error) {
	result = &v1beta1.Role{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("roles").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Roles that match those selectors.
func (c *roles) List(opts v1.ListOptions) (result *v1beta1.RoleList, err error) {
	result = &v1beta1.RoleList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("roles").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested roles.
func (c *roles) Watch(opts v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("roles").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched role.
func (c *roles) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1beta1.Role, err error) {
	result = &v1beta1.Role{}
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
