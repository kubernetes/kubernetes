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

// PodTemplateNamespacer has methods to work with PodTemplate resources in a namespace
type PodTemplateNamespacer interface {
	PodTemplates(namespace string) PodTemplateInterface
}

// PodTemplateInterface has methods to work with PodTemplate resources.
type PodTemplateInterface interface {
	Create(*api.PodTemplate) (*api.PodTemplate, error)
	Update(*api.PodTemplate) (*api.PodTemplate, error)
	Delete(name string, options *api.DeleteOptions) error
	Get(name string) (*api.PodTemplate, error)
	List(opts unversioned.ListOptions) (*api.PodTemplateList, error)
	Watch(opts unversioned.ListOptions) (watch.Interface, error)
}

// podTemplates implements PodTemplateInterface
type podTemplates struct {
	client *LegacyClient
	ns     string
}

// newPodTemplates returns a PodTemplates
func newPodTemplates(c *LegacyClient, namespace string) *podTemplates {
	return &podTemplates{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a podTemplate and creates it.  Returns the server's representation of the podTemplate, and an error, if there is any.
func (c *podTemplates) Create(podTemplate *api.PodTemplate) (result *api.PodTemplate, err error) {
	result = &api.PodTemplate{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("podTemplates").
		Body(podTemplate).
		Do().
		Into(result)
	return
}

// Update takes the representation of a podTemplate and updates it. Returns the server's representation of the podTemplate, and an error, if there is any.
func (c *podTemplates) Update(podTemplate *api.PodTemplate) (result *api.PodTemplate, err error) {
	result = &api.PodTemplate{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("podTemplates").
		Name(podTemplate.Name).
		Body(podTemplate).
		Do().
		Into(result)
	return
}

// Delete takes name of the podTemplate and deletes it. Returns an error if one occurs.
func (c *podTemplates) Delete(name string, options *api.DeleteOptions) error {
	if options == nil {
		return c.client.Delete().Namespace(c.ns).Resource("podTemplates").Name(name).Do().Error()
	}
	body, err := api.Scheme.EncodeToVersion(options, c.client.APIVersion().String())
	if err != nil {
		return err
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("podTemplates").
		Name(name).
		Body(body).
		Do().
		Error()
}

// Get takes name of the podTemplate, and returns the corresponding podTemplate object, and an error if there is any.
func (c *podTemplates) Get(name string) (result *api.PodTemplate, err error) {
	result = &api.PodTemplate{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("podTemplates").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of PodTemplates that match those selectors.
func (c *podTemplates) List(opts unversioned.ListOptions) (result *api.PodTemplateList, err error) {
	result = &api.PodTemplateList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("podTemplates").
		VersionedParams(&opts, api.Scheme).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested podTemplates.
func (c *podTemplates) Watch(opts unversioned.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("podTemplates").
		VersionedParams(&opts, api.Scheme).
		Watch()
}
