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
	watch "k8s.io/kubernetes/pkg/watch"
)

// NamespaceNamespacer has methods to work with Namespace resources in a namespace
type NamespaceNamespacer interface {
	Namespaces(namespace string) NamespaceInterface
}

// NamespaceInterface has methods to work with Namespace resources.
type NamespaceInterface interface {
	Create(*api.Namespace) (*api.Namespace, error)
	Update(*api.Namespace) (*api.Namespace, error)
	Delete(name string, options *api.DeleteOptions) error
	Get(name string) (*api.Namespace, error)
	List(opts unversioned.ListOptions) (*api.NamespaceList, error)
	Watch(opts unversioned.ListOptions) (watch.Interface, error)
}

// namespaces implements NamespaceInterface
type namespaces struct {
	client *LegacyClient
	ns     string
}

// newNamespaces returns a Namespaces
func newNamespaces(c *LegacyClient, namespace string) *namespaces {
	return &namespaces{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a namespace and creates it.  Returns the server's representation of the namespace, and an error, if there is any.
func (c *namespaces) Create(namespace *api.Namespace) (result *api.Namespace, err error) {
	result = &api.Namespace{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("namespaces").
		Body(namespace).
		Do().
		Into(result)
	return
}

// Update takes the representation of a namespace and updates it. Returns the server's representation of the namespace, and an error, if there is any.
func (c *namespaces) Update(namespace *api.Namespace) (result *api.Namespace, err error) {
	result = &api.Namespace{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("namespaces").
		Name(namespace.Name).
		Body(namespace).
		Do().
		Into(result)
	return
}

// Delete takes name of the namespace and deletes it. Returns an error if one occurs.
func (c *namespaces) Delete(name string, options *api.DeleteOptions) error {
	if options == nil {
		return c.client.Delete().Namespace(c.ns).Resource("namespaces").Name(name).Do().Error()
	}
	body, err := api.Scheme.EncodeToVersion(options, c.client.APIVersion().String())
	if err != nil {
		return err
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("namespaces").
		Name(name).
		Body(body).
		Do().
		Error()
}

// Get takes name of the namespace, and returns the corresponding namespace object, and an error if there is any.
func (c *namespaces) Get(name string) (result *api.Namespace, err error) {
	result = &api.Namespace{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("namespaces").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Namespaces that match those selectors.
func (c *namespaces) List(opts unversioned.ListOptions) (result *api.NamespaceList, err error) {
	result = &api.NamespaceList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("namespaces").
		VersionedParams(&opts, api.Scheme).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested namespaces.
func (c *namespaces) Watch(opts unversioned.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("namespaces").
		VersionedParams(&opts, api.Scheme).
		Watch()
}
