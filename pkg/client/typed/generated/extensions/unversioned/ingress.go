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
	api "k8s.io/kubernetes/pkg/api"
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	extensions "k8s.io/kubernetes/pkg/apis/extensions"
	watch "k8s.io/kubernetes/pkg/watch"
)

// IngressNamespacer has methods to work with Ingress resources in a namespace
type IngressNamespacer interface {
	Ingresses(namespace string) IngressInterface
}

// IngressInterface has methods to work with Ingress resources.
type IngressInterface interface {
	Create(*extensions.Ingress) (*extensions.Ingress, error)
	Update(*extensions.Ingress) (*extensions.Ingress, error)
	Delete(name string, options *api.DeleteOptions) error
	Get(name string) (*extensions.Ingress, error)
	List(opts unversioned.ListOptions) (*extensions.IngressList, error)
	Watch(opts unversioned.ListOptions) (watch.Interface, error)
}

// ingresses implements IngressInterface
type ingresses struct {
	client *ExtensionsClient
	ns     string
}

// newIngresses returns a Ingresses
func newIngresses(c *ExtensionsClient, namespace string) *ingresses {
	return &ingresses{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a ingress and creates it.  Returns the server's representation of the ingress, and an error, if there is any.
func (c *ingresses) Create(ingress *extensions.Ingress) (result *extensions.Ingress, err error) {
	result = &extensions.Ingress{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("ingresses").
		Body(ingress).
		Do().
		Into(result)
	return
}

// Update takes the representation of a ingress and updates it. Returns the server's representation of the ingress, and an error, if there is any.
func (c *ingresses) Update(ingress *extensions.Ingress) (result *extensions.Ingress, err error) {
	result = &extensions.Ingress{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("ingresses").
		Name(ingress.Name).
		Body(ingress).
		Do().
		Into(result)
	return
}

// Delete takes name of the ingress and deletes it. Returns an error if one occurs.
func (c *ingresses) Delete(name string, options *api.DeleteOptions) error {
	if options == nil {
		return c.client.Delete().Namespace(c.ns).Resource("ingresses").Name(name).Do().Error()
	}
	body, err := api.Scheme.EncodeToVersion(options, c.client.APIVersion().String())
	if err != nil {
		return err
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("ingresses").
		Name(name).
		Body(body).
		Do().
		Error()
}

// Get takes name of the ingress, and returns the corresponding ingress object, and an error if there is any.
func (c *ingresses) Get(name string) (result *extensions.Ingress, err error) {
	result = &extensions.Ingress{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("ingresses").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Ingresses that match those selectors.
func (c *ingresses) List(opts unversioned.ListOptions) (result *extensions.IngressList, err error) {
	result = &extensions.IngressList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("ingresses").
		VersionedParams(&opts, api.Scheme).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested ingresses.
func (c *ingresses) Watch(opts unversioned.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("ingresses").
		VersionedParams(&opts, api.Scheme).
		Watch()
}
