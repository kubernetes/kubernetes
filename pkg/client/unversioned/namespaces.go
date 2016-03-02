/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/watch"
)

type NamespacesInterface interface {
	Namespaces() NamespaceInterface
}

type NamespaceInterface interface {
	Create(item *api.Namespace) (*api.Namespace, error)
	Get(name string) (result *api.Namespace, err error)
	List(opts api.ListOptions) (*api.NamespaceList, error)
	Delete(name string) error
	Update(item *api.Namespace) (*api.Namespace, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Finalize(item *api.Namespace) (*api.Namespace, error)
	Status(item *api.Namespace) (*api.Namespace, error)
}

// namespaces implements NamespacesInterface
type namespaces struct {
	r *Client
}

// newNamespaces returns a namespaces object.
func newNamespaces(c *Client) *namespaces {
	return &namespaces{r: c}
}

// Create creates a new namespace.
func (c *namespaces) Create(namespace *api.Namespace) (*api.Namespace, error) {
	result := &api.Namespace{}
	err := c.r.Post().Resource("namespaces").Body(namespace).Do().Into(result)
	return result, err
}

// List lists all the namespaces in the cluster.
func (c *namespaces) List(opts api.ListOptions) (*api.NamespaceList, error) {
	result := &api.NamespaceList{}
	err := c.r.Get().
		Resource("namespaces").
		VersionedParams(&opts, api.ParameterCodec).
		Do().Into(result)
	return result, err
}

// Update takes the representation of a namespace to update.  Returns the server's representation of the namespace, and an error, if it occurs.
func (c *namespaces) Update(namespace *api.Namespace) (result *api.Namespace, err error) {
	result = &api.Namespace{}
	err = c.r.Put().Resource("namespaces").Name(namespace.Name).Body(namespace).Do().Into(result)
	return
}

// Finalize takes the representation of a namespace to update.  Returns the server's representation of the namespace, and an error, if it occurs.
func (c *namespaces) Finalize(namespace *api.Namespace) (result *api.Namespace, err error) {
	result = &api.Namespace{}
	if len(namespace.ResourceVersion) == 0 {
		err = fmt.Errorf("invalid update object, missing resource version: %v", namespace)
		return
	}
	err = c.r.Put().Resource("namespaces").Name(namespace.Name).SubResource("finalize").Body(namespace).Do().Into(result)
	return
}

// Status takes the representation of a namespace to update.  Returns the server's representation of the namespace, and an error, if it occurs.
func (c *namespaces) Status(namespace *api.Namespace) (result *api.Namespace, err error) {
	result = &api.Namespace{}
	if len(namespace.ResourceVersion) == 0 {
		err = fmt.Errorf("invalid update object, missing resource version: %v", namespace)
		return
	}
	err = c.r.Put().Resource("namespaces").Name(namespace.Name).SubResource("status").Body(namespace).Do().Into(result)
	return
}

// Get gets an existing namespace
func (c *namespaces) Get(name string) (*api.Namespace, error) {
	result := &api.Namespace{}
	err := c.r.Get().Resource("namespaces").Name(name).Do().Into(result)
	return result, err
}

// Delete deletes an existing namespace.
func (c *namespaces) Delete(name string) error {
	return c.r.Delete().Resource("namespaces").Name(name).Do().Error()
}

// Watch returns a watch.Interface that watches the requested namespaces.
func (c *namespaces) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Resource("namespaces").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}
