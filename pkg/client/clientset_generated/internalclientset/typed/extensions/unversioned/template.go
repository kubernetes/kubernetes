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

// TemplatesGetter has a method to return a TemplateInterface.
// A group's client should implement this interface.
type TemplatesGetter interface {
	Templates(namespace string) TemplateInterface
}

// TemplateInterface has methods to work with Template resources.
type TemplateInterface interface {
	Create(*extensions.Template) (*extensions.Template, error)
	Update(*extensions.Template) (*extensions.Template, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*extensions.Template, error)
	List(opts api.ListOptions) (*extensions.TemplateList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	TemplateExpansion
}

// templates implements TemplateInterface
type templates struct {
	client *ExtensionsClient
	ns     string
}

// newTemplates returns a Templates
func newTemplates(c *ExtensionsClient, namespace string) *templates {
	return &templates{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a template and creates it.  Returns the server's representation of the template, and an error, if there is any.
func (c *templates) Create(template *extensions.Template) (result *extensions.Template, err error) {
	result = &extensions.Template{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("templates").
		Body(template).
		Do().
		Into(result)
	return
}

// Update takes the representation of a template and updates it. Returns the server's representation of the template, and an error, if there is any.
func (c *templates) Update(template *extensions.Template) (result *extensions.Template, err error) {
	result = &extensions.Template{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("templates").
		Name(template.Name).
		Body(template).
		Do().
		Into(result)
	return
}

// Delete takes name of the template and deletes it. Returns an error if one occurs.
func (c *templates) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("templates").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *templates) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("templates").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the template, and returns the corresponding template object, and an error if there is any.
func (c *templates) Get(name string) (result *extensions.Template, err error) {
	result = &extensions.Template{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("templates").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Templates that match those selectors.
func (c *templates) List(opts api.ListOptions) (result *extensions.TemplateList, err error) {
	result = &extensions.TemplateList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("templates").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested templates.
func (c *templates) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("templates").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}
