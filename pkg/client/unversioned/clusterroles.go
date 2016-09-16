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

// ClusterRoles has methods to work with ClusterRole resources in a namespace
type ClusterRoles interface {
	ClusterRoles() ClusterRoleInterface
}

// ClusterRoleInterface has methods to work with ClusterRole resources.
type ClusterRoleInterface interface {
	List(opts api.ListOptions) (*rbac.ClusterRoleList, error)
	Get(name string) (*rbac.ClusterRole, error)
	Delete(name string, options *api.DeleteOptions) error
	Create(clusterRole *rbac.ClusterRole) (*rbac.ClusterRole, error)
	Update(clusterRole *rbac.ClusterRole) (*rbac.ClusterRole, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
}

// clusterRoles implements ClusterRolesNamespacer interface
type clusterRoles struct {
	client *RbacClient
}

// newClusterRoles returns a clusterRoles
func newClusterRoles(c *RbacClient) *clusterRoles {
	return &clusterRoles{
		client: c,
	}
}

// List takes label and field selectors, and returns the list of clusterRoles that match those selectors.
func (c *clusterRoles) List(opts api.ListOptions) (result *rbac.ClusterRoleList, err error) {
	result = &rbac.ClusterRoleList{}
	err = c.client.Get().Resource("clusterroles").VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return
}

// Get takes the name of the clusterRole, and returns the corresponding ClusterRole object, and an error if it occurs
func (c *clusterRoles) Get(name string) (result *rbac.ClusterRole, err error) {
	result = &rbac.ClusterRole{}
	err = c.client.Get().Resource("clusterroles").Name(name).Do().Into(result)
	return
}

// Delete takes the name of the clusterRole and deletes it.  Returns an error if one occurs.
func (c *clusterRoles) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().Resource("clusterroles").Name(name).Body(options).Do().Error()
}

// Create takes the representation of a clusterRole and creates it.  Returns the server's representation of the clusterRole, and an error, if it occurs.
func (c *clusterRoles) Create(clusterRole *rbac.ClusterRole) (result *rbac.ClusterRole, err error) {
	result = &rbac.ClusterRole{}
	err = c.client.Post().Resource("clusterroles").Body(clusterRole).Do().Into(result)
	return
}

// Update takes the representation of a clusterRole and updates it.  Returns the server's representation of the clusterRole, and an error, if it occurs.
func (c *clusterRoles) Update(clusterRole *rbac.ClusterRole) (result *rbac.ClusterRole, err error) {
	result = &rbac.ClusterRole{}
	err = c.client.Put().Resource("clusterroles").Name(clusterRole.Name).Body(clusterRole).Do().Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested clusterRoles.
func (c *clusterRoles) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Resource("clusterroles").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}
