/*
Copyright 2014 Google Inc. All rights reserved.

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

package client

import (
	"errors"
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

type NamespacesInterface interface {
	Namespaces() NamespaceInterface
}

type NamespaceInterface interface {
	Create(item *api.Namespace) (*api.Namespace, error)
	Get(name string) (result *api.Namespace, err error)
	List(selector labels.Selector) (*api.NamespaceList, error)
	Delete(name string) error
	Update(item *api.Namespace) (*api.Namespace, error)
	Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error)
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
func (c *namespaces) List(selector labels.Selector) (*api.NamespaceList, error) {
	result := &api.NamespaceList{}
	err := c.r.Get().Resource("namespaces").SelectorParam(api.LabelSelectorQueryParam(c.r.APIVersion()), selector).Do().Into(result)
	return result, err
}

// Update takes the representation of a namespace to update.  Returns the server's representation of the namespace, and an error, if it occurs.
func (c *namespaces) Update(namespace *api.Namespace) (result *api.Namespace, err error) {
	result = &api.Namespace{}
	if len(namespace.ResourceVersion) == 0 {
		err = fmt.Errorf("invalid update object, missing resource version: %v", namespace)
		return
	}
	err = c.r.Put().Resource("namespaces").Name(namespace.Name).Body(namespace).Do().Into(result)
	return
}

// Get gets an existing namespace
func (c *namespaces) Get(name string) (*api.Namespace, error) {
	if len(name) == 0 {
		return nil, errors.New("name is required parameter to Get")
	}

	result := &api.Namespace{}
	err := c.r.Get().Resource("namespaces").Name(name).Do().Into(result)
	return result, err
}

// Delete deletes an existing namespace.
func (c *namespaces) Delete(name string) error {
	return c.r.Delete().Resource("namespaces").Name(name).Do().Error()
}

// Watch returns a watch.Interface that watches the requested namespaces.
func (c *namespaces) Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Resource("namespaces").
		Param("resourceVersion", resourceVersion).
		SelectorParam(api.LabelSelectorQueryParam(c.r.APIVersion()), label).
		SelectorParam(api.FieldSelectorQueryParam(c.r.APIVersion()), field).
		Watch()
}
