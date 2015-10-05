/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/apis/experimental"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// IngressNamespacer has methods to work with Ingress resources in a namespace
type IngressNamespacer interface {
	Ingress(namespace string) IngressInterface
}

// IngressInterface exposes methods to work on Ingress resources.
type IngressInterface interface {
	List(label labels.Selector, field fields.Selector) (*experimental.IngressList, error)
	Get(name string) (*experimental.Ingress, error)
	Create(ingress *experimental.Ingress) (*experimental.Ingress, error)
	Update(ingress *experimental.Ingress) (*experimental.Ingress, error)
	Delete(name string, options *api.DeleteOptions) error
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
	UpdateStatus(ingress *experimental.Ingress) (*experimental.Ingress, error)
}

// ingress implements IngressNamespacer interface
type ingress struct {
	r  *ExperimentalClient
	ns string
}

// newIngress returns a ingress
func newIngress(c *ExperimentalClient, namespace string) *ingress {
	return &ingress{c, namespace}
}

// List returns a list of ingress that match the label and field selectors.
func (c *ingress) List(label labels.Selector, field fields.Selector) (result *experimental.IngressList, err error) {
	result = &experimental.IngressList{}
	err = c.r.Get().Namespace(c.ns).Resource("ingress").LabelsSelectorParam(label).FieldsSelectorParam(field).Do().Into(result)
	return
}

// Get returns information about a particular ingress.
func (c *ingress) Get(name string) (result *experimental.Ingress, err error) {
	result = &experimental.Ingress{}
	err = c.r.Get().Namespace(c.ns).Resource("ingress").Name(name).Do().Into(result)
	return
}

// Create creates a new ingress.
func (c *ingress) Create(ingress *experimental.Ingress) (result *experimental.Ingress, err error) {
	result = &experimental.Ingress{}
	err = c.r.Post().Namespace(c.ns).Resource("ingress").Body(ingress).Do().Into(result)
	return
}

// Update updates an existing ingress.
func (c *ingress) Update(ingress *experimental.Ingress) (result *experimental.Ingress, err error) {
	result = &experimental.Ingress{}
	err = c.r.Put().Namespace(c.ns).Resource("ingress").Name(ingress.Name).Body(ingress).Do().Into(result)
	return
}

// Delete deletes a ingress, returns error if one occurs.
func (c *ingress) Delete(name string, options *api.DeleteOptions) (err error) {
	if options == nil {
		return c.r.Delete().Namespace(c.ns).Resource("ingress").Name(name).Do().Error()
	}

	body, err := api.Scheme.EncodeToVersion(options, c.r.APIVersion())
	if err != nil {
		return err
	}
	return c.r.Delete().Namespace(c.ns).Resource("ingress").Name(name).Body(body).Do().Error()
}

// Watch returns a watch.Interface that watches the requested ingress.
func (c *ingress) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("ingress").
		Param("resourceVersion", resourceVersion).
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Watch()
}

// UpdateStatus takes the name of the ingress and the new status.  Returns the server's representation of the ingress, and an error, if it occurs.
func (c *ingress) UpdateStatus(ingress *experimental.Ingress) (result *experimental.Ingress, err error) {
	result = &experimental.Ingress{}
	err = c.r.Put().Namespace(c.ns).Resource("ingress").Name(ingress.Name).SubResource("status").Body(ingress).Do().Into(result)
	return
}
