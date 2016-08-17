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

// ClusterRoleBindings has methods to work with ClusterRoleBinding resources in a namespace
type ClusterRoleBindings interface {
	ClusterRoleBindings() ClusterRoleBindingInterface
}

// ClusterRoleBindingInterface has methods to work with ClusterRoleBinding resources.
type ClusterRoleBindingInterface interface {
	List(opts api.ListOptions) (*rbac.ClusterRoleBindingList, error)
	Get(name string) (*rbac.ClusterRoleBinding, error)
	Delete(name string, options *api.DeleteOptions) error
	Create(clusterRoleBinding *rbac.ClusterRoleBinding) (*rbac.ClusterRoleBinding, error)
	Update(clusterRoleBinding *rbac.ClusterRoleBinding) (*rbac.ClusterRoleBinding, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
}

// clusterRoleBindings implements ClusterRoleBindingsNamespacer interface
type clusterRoleBindings struct {
	client *RbacClient
}

// newClusterRoleBindings returns a clusterRoleBindings
func newClusterRoleBindings(c *RbacClient) *clusterRoleBindings {
	return &clusterRoleBindings{
		client: c,
	}
}

// List takes label and field selectors, and returns the list of clusterRoleBindings that match those selectors.
func (c *clusterRoleBindings) List(opts api.ListOptions) (result *rbac.ClusterRoleBindingList, err error) {
	result = &rbac.ClusterRoleBindingList{}
	err = c.client.Get().Resource("clusterrolebindings").VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return
}

// Get takes the name of the clusterRoleBinding, and returns the corresponding ClusterRoleBinding object, and an error if it occurs
func (c *clusterRoleBindings) Get(name string) (result *rbac.ClusterRoleBinding, err error) {
	result = &rbac.ClusterRoleBinding{}
	err = c.client.Get().Resource("clusterrolebindings").Name(name).Do().Into(result)
	return
}

// Delete takes the name of the clusterRoleBinding and deletes it.  Returns an error if one occurs.
func (c *clusterRoleBindings) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().Resource("clusterrolebindings").Name(name).Body(options).Do().Error()
}

// Create takes the representation of a clusterRoleBinding and creates it.  Returns the server's representation of the clusterRoleBinding, and an error, if it occurs.
func (c *clusterRoleBindings) Create(clusterRoleBinding *rbac.ClusterRoleBinding) (result *rbac.ClusterRoleBinding, err error) {
	result = &rbac.ClusterRoleBinding{}
	err = c.client.Post().Resource("clusterrolebindings").Body(clusterRoleBinding).Do().Into(result)
	return
}

// Update takes the representation of a clusterRoleBinding and updates it.  Returns the server's representation of the clusterRoleBinding, and an error, if it occurs.
func (c *clusterRoleBindings) Update(clusterRoleBinding *rbac.ClusterRoleBinding) (result *rbac.ClusterRoleBinding, err error) {
	result = &rbac.ClusterRoleBinding{}
	err = c.client.Put().Resource("clusterrolebindings").Name(clusterRoleBinding.Name).Body(clusterRoleBinding).Do().Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested clusterRoleBindings.
func (c *clusterRoleBindings) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Resource("clusterrolebindings").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}
