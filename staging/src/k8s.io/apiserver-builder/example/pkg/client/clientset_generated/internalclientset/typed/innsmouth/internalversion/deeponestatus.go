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
	innsmouth "k8s.io/apiserver-builder/example/pkg/apis/innsmouth"
	scheme "k8s.io/apiserver-builder/example/pkg/client/clientset_generated/internalclientset/scheme"
	rest "k8s.io/client-go/rest"
)

// DeepOneStatusesGetter has a method to return a DeepOneStatusInterface.
// A group's client should implement this interface.
type DeepOneStatusesGetter interface {
	DeepOneStatuses(namespace string) DeepOneStatusInterface
}

// DeepOneStatusInterface has methods to work with DeepOneStatus resources.
type DeepOneStatusInterface interface {
	Create(*innsmouth.DeepOneStatus) (*innsmouth.DeepOneStatus, error)
	Update(*innsmouth.DeepOneStatus) (*innsmouth.DeepOneStatus, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*innsmouth.DeepOneStatus, error)
	List(opts v1.ListOptions) (*innsmouth.DeepOneStatusList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *innsmouth.DeepOneStatus, err error)
	DeepOneStatusExpansion
}

// deepOneStatuses implements DeepOneStatusInterface
type deepOneStatuses struct {
	client rest.Interface
	ns     string
}

// newDeepOneStatuses returns a DeepOneStatuses
func newDeepOneStatuses(c *InnsmouthClient, namespace string) *deepOneStatuses {
	return &deepOneStatuses{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a deepOneStatus and creates it.  Returns the server's representation of the deepOneStatus, and an error, if there is any.
func (c *deepOneStatuses) Create(deepOneStatus *innsmouth.DeepOneStatus) (result *innsmouth.DeepOneStatus, err error) {
	result = &innsmouth.DeepOneStatus{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("deeponestatuses").
		Body(deepOneStatus).
		Do().
		Into(result)
	return
}

// Update takes the representation of a deepOneStatus and updates it. Returns the server's representation of the deepOneStatus, and an error, if there is any.
func (c *deepOneStatuses) Update(deepOneStatus *innsmouth.DeepOneStatus) (result *innsmouth.DeepOneStatus, err error) {
	result = &innsmouth.DeepOneStatus{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("deeponestatuses").
		Name(deepOneStatus.Name).
		Body(deepOneStatus).
		Do().
		Into(result)
	return
}

// Delete takes name of the deepOneStatus and deletes it. Returns an error if one occurs.
func (c *deepOneStatuses) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("deeponestatuses").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *deepOneStatuses) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("deeponestatuses").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the deepOneStatus, and returns the corresponding deepOneStatus object, and an error if there is any.
func (c *deepOneStatuses) Get(name string, options v1.GetOptions) (result *innsmouth.DeepOneStatus, err error) {
	result = &innsmouth.DeepOneStatus{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("deeponestatuses").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of DeepOneStatuses that match those selectors.
func (c *deepOneStatuses) List(opts v1.ListOptions) (result *innsmouth.DeepOneStatusList, err error) {
	result = &innsmouth.DeepOneStatusList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("deeponestatuses").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested deepOneStatuses.
func (c *deepOneStatuses) Watch(opts v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("deeponestatuses").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched deepOneStatus.
func (c *deepOneStatuses) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *innsmouth.DeepOneStatus, err error) {
	result = &innsmouth.DeepOneStatus{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("deeponestatuses").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
