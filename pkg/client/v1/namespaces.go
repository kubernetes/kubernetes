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

package v1

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

type NamespacesInterface interface {
	Namespaces() NamespaceInterface
}

type NamespaceInterface interface {
	Create(item *v1.Namespace) (*v1.Namespace, error)
	Get(name string) (result *v1.Namespace, err error)
	List(label labels.Selector, field fields.Selector) (*v1.NamespaceList, error)
	Delete(name string) error
	Update(item *v1.Namespace) (*v1.Namespace, error)
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
	Finalize(item *v1.Namespace) (*v1.Namespace, error)
	Status(item *v1.Namespace) (*v1.Namespace, error)
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
func (c *namespaces) Create(namespace *v1.Namespace) (*v1.Namespace, error) {
	result := &v1.Namespace{}
	err := c.r.Post().Resource("namespaces").Body(namespace).Do().Into(result)
	return result, err
}

// List lists all the namespaces in the cluster.
func (c *namespaces) List(label labels.Selector, field fields.Selector) (*v1.NamespaceList, error) {
	result := &v1.NamespaceList{}
	err := c.r.Get().
		Resource("namespaces").
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Do().Into(result)
	return result, err
}

// Update takes the representation of a namespace to update.  Returns the server's representation of the namespace, and an error, if it occurs.
func (c *namespaces) Update(namespace *v1.Namespace) (result *v1.Namespace, err error) {
	result = &v1.Namespace{}
	if len(namespace.ResourceVersion) == 0 {
		err = fmt.Errorf("invalid update object, missing resource version: %v", namespace)
		return
	}
	err = c.r.Put().Resource("namespaces").Name(namespace.Name).Body(namespace).Do().Into(result)
	return
}

// Finalize takes the representation of a namespace to update.  Returns the server's representation of the namespace, and an error, if it occurs.
func (c *namespaces) Finalize(namespace *v1.Namespace) (result *v1.Namespace, err error) {
	result = &v1.Namespace{}
	if len(namespace.ResourceVersion) == 0 {
		err = fmt.Errorf("invalid update object, missing resource version: %v", namespace)
		return
	}
	err = c.r.Put().Resource("namespaces").Name(namespace.Name).SubResource("finalize").Body(namespace).Do().Into(result)
	return
}

// Status takes the representation of a namespace to update.  Returns the server's representation of the namespace, and an error, if it occurs.
func (c *namespaces) Status(namespace *v1.Namespace) (result *v1.Namespace, err error) {
	result = &v1.Namespace{}
	if len(namespace.ResourceVersion) == 0 {
		err = fmt.Errorf("invalid update object, missing resource version: %v", namespace)
		return
	}
	err = c.r.Put().Resource("namespaces").Name(namespace.Name).SubResource("status").Body(namespace).Do().Into(result)
	return
}

// Get gets an existing namespace
func (c *namespaces) Get(name string) (*v1.Namespace, error) {
	result := &v1.Namespace{}
	err := c.r.Get().Resource("namespaces").Name(name).Do().Into(result)
	return result, err
}

// Delete deletes an existing namespace.
func (c *namespaces) Delete(name string) error {
	return c.r.Delete().Resource("namespaces").Name(name).Do().Error()
}

// Watch returns a watch.Interface that watches the requested namespaces.
func (c *namespaces) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Resource("namespaces").
		Param("resourceVersion", resourceVersion).
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Watch()
}
