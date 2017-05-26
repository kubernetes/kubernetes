/*
Copyright 2017 The Kubernetes Authors.

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

package internalversion

import (
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	rest "k8s.io/client-go/rest"
	admissionregistration "k8s.io/kubernetes/pkg/apis/admissionregistration"
	scheme "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/scheme"
)

// ExternalAdmissionHookConfigurationsGetter has a method to return a ExternalAdmissionHookConfigurationInterface.
// A group's client should implement this interface.
type ExternalAdmissionHookConfigurationsGetter interface {
	ExternalAdmissionHookConfigurations(namespace string) ExternalAdmissionHookConfigurationInterface
}

// ExternalAdmissionHookConfigurationInterface has methods to work with ExternalAdmissionHookConfiguration resources.
type ExternalAdmissionHookConfigurationInterface interface {
	Create(*admissionregistration.ExternalAdmissionHookConfiguration) (*admissionregistration.ExternalAdmissionHookConfiguration, error)
	Update(*admissionregistration.ExternalAdmissionHookConfiguration) (*admissionregistration.ExternalAdmissionHookConfiguration, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*admissionregistration.ExternalAdmissionHookConfiguration, error)
	List(opts v1.ListOptions) (*admissionregistration.ExternalAdmissionHookConfigurationList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *admissionregistration.ExternalAdmissionHookConfiguration, err error)
	ExternalAdmissionHookConfigurationExpansion
}

// externalAdmissionHookConfigurations implements ExternalAdmissionHookConfigurationInterface
type externalAdmissionHookConfigurations struct {
	client rest.Interface
	ns     string
}

// newExternalAdmissionHookConfigurations returns a ExternalAdmissionHookConfigurations
func newExternalAdmissionHookConfigurations(c *AdmissionregistrationClient, namespace string) *externalAdmissionHookConfigurations {
	return &externalAdmissionHookConfigurations{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a externalAdmissionHookConfiguration and creates it.  Returns the server's representation of the externalAdmissionHookConfiguration, and an error, if there is any.
func (c *externalAdmissionHookConfigurations) Create(externalAdmissionHookConfiguration *admissionregistration.ExternalAdmissionHookConfiguration) (result *admissionregistration.ExternalAdmissionHookConfiguration, err error) {
	result = &admissionregistration.ExternalAdmissionHookConfiguration{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("externaladmissionhookconfigurations").
		Body(externalAdmissionHookConfiguration).
		Do().
		Into(result)
	return
}

// Update takes the representation of a externalAdmissionHookConfiguration and updates it. Returns the server's representation of the externalAdmissionHookConfiguration, and an error, if there is any.
func (c *externalAdmissionHookConfigurations) Update(externalAdmissionHookConfiguration *admissionregistration.ExternalAdmissionHookConfiguration) (result *admissionregistration.ExternalAdmissionHookConfiguration, err error) {
	result = &admissionregistration.ExternalAdmissionHookConfiguration{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("externaladmissionhookconfigurations").
		Name(externalAdmissionHookConfiguration.Name).
		Body(externalAdmissionHookConfiguration).
		Do().
		Into(result)
	return
}

// Delete takes name of the externalAdmissionHookConfiguration and deletes it. Returns an error if one occurs.
func (c *externalAdmissionHookConfigurations) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("externaladmissionhookconfigurations").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *externalAdmissionHookConfigurations) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("externaladmissionhookconfigurations").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the externalAdmissionHookConfiguration, and returns the corresponding externalAdmissionHookConfiguration object, and an error if there is any.
func (c *externalAdmissionHookConfigurations) Get(name string, options v1.GetOptions) (result *admissionregistration.ExternalAdmissionHookConfiguration, err error) {
	result = &admissionregistration.ExternalAdmissionHookConfiguration{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("externaladmissionhookconfigurations").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ExternalAdmissionHookConfigurations that match those selectors.
func (c *externalAdmissionHookConfigurations) List(opts v1.ListOptions) (result *admissionregistration.ExternalAdmissionHookConfigurationList, err error) {
	result = &admissionregistration.ExternalAdmissionHookConfigurationList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("externaladmissionhookconfigurations").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested externalAdmissionHookConfigurations.
func (c *externalAdmissionHookConfigurations) Watch(opts v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("externaladmissionhookconfigurations").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched externalAdmissionHookConfiguration.
func (c *externalAdmissionHookConfigurations) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *admissionregistration.ExternalAdmissionHookConfiguration, err error) {
	result = &admissionregistration.ExternalAdmissionHookConfiguration{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("externaladmissionhookconfigurations").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
