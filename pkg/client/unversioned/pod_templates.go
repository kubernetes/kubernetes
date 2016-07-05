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
	"k8s.io/kubernetes/pkg/watch"
)

// PodTemplatesNamespacer has methods to work with PodTemplate resources in a namespace
type PodTemplatesNamespacer interface {
	PodTemplates(namespace string) PodTemplateInterface
}

// PodTemplateInterface has methods to work with PodTemplate resources.
type PodTemplateInterface interface {
	List(opts api.ListOptions) (*api.PodTemplateList, error)
	Get(name string) (*api.PodTemplate, error)
	Delete(name string, options *api.DeleteOptions) error
	Create(podTemplate *api.PodTemplate) (*api.PodTemplate, error)
	Update(podTemplate *api.PodTemplate) (*api.PodTemplate, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
}

// podTemplates implements PodTemplatesNamespacer interface
type podTemplates struct {
	r  *Client
	ns string
}

// newPodTemplates returns a podTemplates
func newPodTemplates(c *Client, namespace string) *podTemplates {
	return &podTemplates{
		r:  c,
		ns: namespace,
	}
}

// List takes label and field selectors, and returns the list of podTemplates that match those selectors.
func (c *podTemplates) List(opts api.ListOptions) (result *api.PodTemplateList, err error) {
	result = &api.PodTemplateList{}
	err = c.r.Get().Namespace(c.ns).Resource("podTemplates").VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return
}

// Get takes the name of the podTemplate, and returns the corresponding PodTemplate object, and an error if it occurs
func (c *podTemplates) Get(name string) (result *api.PodTemplate, err error) {
	result = &api.PodTemplate{}
	err = c.r.Get().Namespace(c.ns).Resource("podTemplates").Name(name).Do().Into(result)
	return
}

// Delete takes the name of the podTemplate, and returns an error if one occurs
func (c *podTemplates) Delete(name string, options *api.DeleteOptions) error {
	return c.r.Delete().Namespace(c.ns).Resource("podTemplates").Name(name).Body(options).Do().Error()
}

// Create takes the representation of a podTemplate.  Returns the server's representation of the podTemplate, and an error, if it occurs.
func (c *podTemplates) Create(podTemplate *api.PodTemplate) (result *api.PodTemplate, err error) {
	result = &api.PodTemplate{}
	err = c.r.Post().Namespace(c.ns).Resource("podTemplates").Body(podTemplate).Do().Into(result)
	return
}

// Update takes the representation of a podTemplate to update.  Returns the server's representation of the podTemplate, and an error, if it occurs.
func (c *podTemplates) Update(podTemplate *api.PodTemplate) (result *api.PodTemplate, err error) {
	result = &api.PodTemplate{}
	err = c.r.Put().Namespace(c.ns).Resource("podTemplates").Name(podTemplate.Name).Body(podTemplate).Do().Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested podTemplates.
func (c *podTemplates) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("podTemplates").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}
