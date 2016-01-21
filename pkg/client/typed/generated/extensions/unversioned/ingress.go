/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	api "k8s.io/kubernetes/pkg/api"
	extensions "k8s.io/kubernetes/pkg/apis/extensions"
	watch "k8s.io/kubernetes/pkg/watch"
)

// IngressGetter has a method to return a IngressInterface.
// A group's client should implement this interface.
type IngressGetter interface {
	Ingress(namespace string) IngressInterface
}

// IngressInterface has methods to work with Ingress resources.
type IngressInterface interface {
	Create(*extensions.Ingress) (*extensions.Ingress, error)
	Update(*extensions.Ingress) (*extensions.Ingress, error)
	UpdateStatus(*extensions.Ingress) (*extensions.Ingress, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*extensions.Ingress, error)
	List(opts api.ListOptions) (*extensions.IngressList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	IngressExpansion
}

// ingress implements IngressInterface
type ingress struct {
	client *ExtensionsClient
	ns     string
}

// newIngress returns a Ingress
func newIngress(c *ExtensionsClient, namespace string) *ingress {
	return &ingress{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a ingress and creates it.  Returns the server's representation of the ingress, and an error, if there is any.
func (c *ingress) Create(ingress *extensions.Ingress) (result *extensions.Ingress, err error) {
	result = &extensions.Ingress{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("ingress").
		Body(ingress).
		Do().
		Into(result)
	return
}

// Update takes the representation of a ingress and updates it. Returns the server's representation of the ingress, and an error, if there is any.
func (c *ingress) Update(ingress *extensions.Ingress) (result *extensions.Ingress, err error) {
	result = &extensions.Ingress{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("ingress").
		Name(ingress.Name).
		Body(ingress).
		Do().
		Into(result)
	return
}

func (c *ingress) UpdateStatus(ingress *extensions.Ingress) (result *extensions.Ingress, err error) {
	result = &extensions.Ingress{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("ingress").
		Name(ingress.Name).
		SubResource("status").
		Body(ingress).
		Do().
		Into(result)
	return
}

// Delete takes name of the ingress and deletes it. Returns an error if one occurs.
func (c *ingress) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("ingress").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *ingress) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("ingress").
		VersionedParams(&listOptions, api.Scheme).
		Body(options).
		Do().
		Error()
}

// Get takes name of the ingress, and returns the corresponding ingress object, and an error if there is any.
func (c *ingress) Get(name string) (result *extensions.Ingress, err error) {
	result = &extensions.Ingress{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("ingress").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Ingress that match those selectors.
func (c *ingress) List(opts api.ListOptions) (result *extensions.IngressList, err error) {
	result = &extensions.IngressList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("ingress").
		VersionedParams(&opts, api.Scheme).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested ingress.
func (c *ingress) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("ingress").
		VersionedParams(&opts, api.Scheme).
		Watch()
}
