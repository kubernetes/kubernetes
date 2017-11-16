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

package v1alpha1

import (
	v1alpha1 "k8s.io/api/admissionregistration/v1alpha1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	scheme "k8s.io/client-go/kubernetes/scheme"
	rest "k8s.io/client-go/rest"
)

// InitializerConfigurationsGetter has a method to return a InitializerConfigurationInterface.
// A group's client should implement this interface.
type InitializerConfigurationsGetter interface {
	InitializerConfigurations() InitializerConfigurationInterface
}

// InitializerConfigurationInterface has methods to work with InitializerConfiguration resources.
type InitializerConfigurationInterface interface {
	Create(*v1alpha1.InitializerConfiguration) (*v1alpha1.InitializerConfiguration, error)
	Update(*v1alpha1.InitializerConfiguration) (*v1alpha1.InitializerConfiguration, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*v1alpha1.InitializerConfiguration, error)
	List(opts v1.ListOptions) (*v1alpha1.InitializerConfigurationList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1alpha1.InitializerConfiguration, err error)
	InitializerConfigurationExpansion
}

// initializerConfigurations implements InitializerConfigurationInterface
type initializerConfigurations struct {
	client rest.Interface
}

// newInitializerConfigurations returns a InitializerConfigurations
func newInitializerConfigurations(c *AdmissionregistrationV1alpha1Client) *initializerConfigurations {
	return &initializerConfigurations{
		client: c.RESTClient(),
	}
}

// Get takes name of the initializerConfiguration, and returns the corresponding initializerConfiguration object, and an error if there is any.
func (c *initializerConfigurations) Get(name string, options v1.GetOptions) (result *v1alpha1.InitializerConfiguration, err error) {
	result = &v1alpha1.InitializerConfiguration{}
	err = c.client.Get().
		Resource("initializerconfigurations").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of InitializerConfigurations that match those selectors.
func (c *initializerConfigurations) List(opts v1.ListOptions) (result *v1alpha1.InitializerConfigurationList, err error) {
	result = &v1alpha1.InitializerConfigurationList{}
	err = c.client.Get().
		Resource("initializerconfigurations").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested initializerConfigurations.
func (c *initializerConfigurations) Watch(opts v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Resource("initializerconfigurations").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}

// Create takes the representation of a initializerConfiguration and creates it.  Returns the server's representation of the initializerConfiguration, and an error, if there is any.
func (c *initializerConfigurations) Create(initializerConfiguration *v1alpha1.InitializerConfiguration) (result *v1alpha1.InitializerConfiguration, err error) {
	result = &v1alpha1.InitializerConfiguration{}
	err = c.client.Post().
		Resource("initializerconfigurations").
		Body(initializerConfiguration).
		Do().
		Into(result)
	return
}

// Update takes the representation of a initializerConfiguration and updates it. Returns the server's representation of the initializerConfiguration, and an error, if there is any.
func (c *initializerConfigurations) Update(initializerConfiguration *v1alpha1.InitializerConfiguration) (result *v1alpha1.InitializerConfiguration, err error) {
	result = &v1alpha1.InitializerConfiguration{}
	err = c.client.Put().
		Resource("initializerconfigurations").
		Name(initializerConfiguration.Name).
		Body(initializerConfiguration).
		Do().
		Into(result)
	return
}

// Delete takes name of the initializerConfiguration and deletes it. Returns an error if one occurs.
func (c *initializerConfigurations) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Resource("initializerconfigurations").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *initializerConfigurations) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Resource("initializerconfigurations").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Patch applies the patch and returns the patched initializerConfiguration.
func (c *initializerConfigurations) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1alpha1.InitializerConfiguration, err error) {
	result = &v1alpha1.InitializerConfiguration{}
	err = c.client.Patch(pt).
		Resource("initializerconfigurations").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
