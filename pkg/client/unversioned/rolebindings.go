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

// RoleBindingsNamespacer has methods to work with RoleBinding resources in a namespace
type RoleBindingsNamespacer interface {
	RoleBindings(namespace string) RoleBindingInterface
}

// RoleBindingInterface has methods to work with RoleBinding resources.
type RoleBindingInterface interface {
	List(opts api.ListOptions) (*rbac.RoleBindingList, error)
	Get(name string) (*rbac.RoleBinding, error)
	Delete(name string, options *api.DeleteOptions) error
	Create(roleBinding *rbac.RoleBinding) (*rbac.RoleBinding, error)
	Update(roleBinding *rbac.RoleBinding) (*rbac.RoleBinding, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
}

// roleBindings implements RoleBindingsNamespacer interface
type roleBindings struct {
	client *RbacClient
	ns     string
}

// newRoleBindings returns a roleBindings
func newRoleBindings(c *RbacClient, namespace string) *roleBindings {
	return &roleBindings{
		client: c,
		ns:     namespace,
	}
}

// List takes label and field selectors, and returns the list of roleBindings that match those selectors.
func (c *roleBindings) List(opts api.ListOptions) (result *rbac.RoleBindingList, err error) {
	result = &rbac.RoleBindingList{}
	err = c.client.Get().Namespace(c.ns).Resource("rolebindings").VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return
}

// Get takes the name of the roleBinding, and returns the corresponding RoleBinding object, and an error if it occurs
func (c *roleBindings) Get(name string) (result *rbac.RoleBinding, err error) {
	result = &rbac.RoleBinding{}
	err = c.client.Get().Namespace(c.ns).Resource("rolebindings").Name(name).Do().Into(result)
	return
}

// Delete takes the name of the roleBinding and deletes it.  Returns an error if one occurs.
func (c *roleBindings) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().Namespace(c.ns).Resource("rolebindings").Name(name).Body(options).Do().Error()
}

// Create takes the representation of a roleBinding and creates it.  Returns the server's representation of the roleBinding, and an error, if it occurs.
func (c *roleBindings) Create(roleBinding *rbac.RoleBinding) (result *rbac.RoleBinding, err error) {
	result = &rbac.RoleBinding{}
	err = c.client.Post().Namespace(c.ns).Resource("rolebindings").Body(roleBinding).Do().Into(result)
	return
}

// Update takes the representation of a roleBinding and updates it.  Returns the server's representation of the roleBinding, and an error, if it occurs.
func (c *roleBindings) Update(roleBinding *rbac.RoleBinding) (result *rbac.RoleBinding, err error) {
	result = &rbac.RoleBinding{}
	err = c.client.Put().Namespace(c.ns).Resource("rolebindings").Name(roleBinding.Name).Body(roleBinding).Do().Into(result)
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
