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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/watch"
)

// RolesNamespacer has methods to work with Role resources in a namespace
type RolesNamespacer interface {
	Roles(namespace string) RoleInterface
}

// RoleInterface has methods to work with Role resources.
type RoleInterface interface {
	List(opts api.ListOptions) (*rbac.RoleList, error)
	Get(name string) (*rbac.Role, error)
	Delete(name string, options *api.DeleteOptions) error
	Create(role *rbac.Role) (*rbac.Role, error)
	Update(role *rbac.Role) (*rbac.Role, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
}

// roles implements RolesNamespacer interface
type roles struct {
	client *RbacClient
	ns     string
}

// newRoles returns a roles
func newRoles(c *RbacClient, namespace string) *roles {
	return &roles{
		client: c,
		ns:     namespace,
	}
}

// List takes label and field selectors, and returns the list of roles that match those selectors.
func (c *roles) List(opts api.ListOptions) (result *rbac.RoleList, err error) {
	result = &rbac.RoleList{}
	err = c.client.Get().Namespace(c.ns).Resource("roles").VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return
}

// Get takes the name of the role, and returns the corresponding Role object, and an error if it occurs
func (c *roles) Get(name string) (result *rbac.Role, err error) {
	result = &rbac.Role{}
	err = c.client.Get().Namespace(c.ns).Resource("roles").Name(name).Do().Into(result)
	return
}

// Delete takes the name of the role and deletes it.  Returns an error if one occurs.
func (c *roles) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().Namespace(c.ns).Resource("roles").Name(name).Body(options).Do().Error()
}

// Create takes the representation of a role and creates it.  Returns the server's representation of the role, and an error, if it occurs.
func (c *roles) Create(role *rbac.Role) (result *rbac.Role, err error) {
	result = &rbac.Role{}
	err = c.client.Post().Namespace(c.ns).Resource("roles").Body(role).Do().Into(result)
	return
}

// Update takes the representation of a role and updates it.  Returns the server's representation of the role, and an error, if it occurs.
func (c *roles) Update(role *rbac.Role) (result *rbac.Role, err error) {
	result = &rbac.Role{}
	err = c.client.Put().Namespace(c.ns).Resource("roles").Name(role.Name).Body(role).Do().Into(result)
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
