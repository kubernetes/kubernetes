/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/watch"
)

// IngressNamespacer has methods to work with Ingress resources in a namespace
type IngressNamespacer interface {
	Ingress(namespace string) IngressInterface
}

// IngressInterface exposes methods to work on Ingress resources.
type IngressInterface interface {
	List(opts api.ListOptions) (*extensions.IngressList, error)
	Get(name string) (*extensions.Ingress, error)
	Create(ingress *extensions.Ingress) (*extensions.Ingress, error)
	Update(ingress *extensions.Ingress) (*extensions.Ingress, error)
	Delete(name string, options *api.DeleteOptions) error
	Watch(opts api.ListOptions) (watch.Interface, error)
	UpdateStatus(ingress *extensions.Ingress) (*extensions.Ingress, error)
}

// ingress implements IngressNamespacer interface
type ingress struct {
	r  *ExtensionsClient
	ns string
}

// newIngress returns a ingress
func newIngress(c *ExtensionsClient, namespace string) *ingress {
	return &ingress{c, namespace}
}

// List returns a list of ingress that match the label and field selectors.
func (c *ingress) List(opts api.ListOptions) (result *extensions.IngressList, err error) {
	result = &extensions.IngressList{}
	err = c.r.Get().Namespace(c.ns).Resource("ingresses").VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return
}

// Get returns information about a particular ingress.
func (c *ingress) Get(name string) (result *extensions.Ingress, err error) {
	result = &extensions.Ingress{}
	err = c.r.Get().Namespace(c.ns).Resource("ingresses").Name(name).Do().Into(result)
	return
}

// Create creates a new ingress.
func (c *ingress) Create(ingress *extensions.Ingress) (result *extensions.Ingress, err error) {
	result = &extensions.Ingress{}
	err = c.r.Post().Namespace(c.ns).Resource("ingresses").Body(ingress).Do().Into(result)
	return
}

// Update updates an existing ingress.
func (c *ingress) Update(ingress *extensions.Ingress) (result *extensions.Ingress, err error) {
	result = &extensions.Ingress{}
	err = c.r.Put().Namespace(c.ns).Resource("ingresses").Name(ingress.Name).Body(ingress).Do().Into(result)
	return
}

// Delete deletes a ingress, returns error if one occurs.
func (c *ingress) Delete(name string, options *api.DeleteOptions) (err error) {
	return c.r.Delete().Namespace(c.ns).Resource("ingresses").Name(name).Body(options).Do().Error()
}

// Watch returns a watch.Interface that watches the requested ingress.
func (c *ingress) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("ingresses").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// UpdateStatus takes the name of the ingress and the new status.  Returns the server's representation of the ingress, and an error, if it occurs.
func (c *ingress) UpdateStatus(ingress *extensions.Ingress) (result *extensions.Ingress, err error) {
	result = &extensions.Ingress{}
	err = c.r.Put().Namespace(c.ns).Resource("ingresses").Name(ingress.Name).SubResource("status").Body(ingress).Do().Into(result)
	return
}
