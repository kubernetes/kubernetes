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

// DeepOneSpecsGetter has a method to return a DeepOneSpecInterface.
// A group's client should implement this interface.
type DeepOneSpecsGetter interface {
	DeepOneSpecs(namespace string) DeepOneSpecInterface
}

// DeepOneSpecInterface has methods to work with DeepOneSpec resources.
type DeepOneSpecInterface interface {
	Create(*innsmouth.DeepOneSpec) (*innsmouth.DeepOneSpec, error)
	Update(*innsmouth.DeepOneSpec) (*innsmouth.DeepOneSpec, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*innsmouth.DeepOneSpec, error)
	List(opts v1.ListOptions) (*innsmouth.DeepOneSpecList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *innsmouth.DeepOneSpec, err error)
	DeepOneSpecExpansion
}

// deepOneSpecs implements DeepOneSpecInterface
type deepOneSpecs struct {
	client rest.Interface
	ns     string
}

// newDeepOneSpecs returns a DeepOneSpecs
func newDeepOneSpecs(c *InnsmouthClient, namespace string) *deepOneSpecs {
	return &deepOneSpecs{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a deepOneSpec and creates it.  Returns the server's representation of the deepOneSpec, and an error, if there is any.
func (c *deepOneSpecs) Create(deepOneSpec *innsmouth.DeepOneSpec) (result *innsmouth.DeepOneSpec, err error) {
	result = &innsmouth.DeepOneSpec{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("deeponespecs").
		Body(deepOneSpec).
		Do().
		Into(result)
	return
}

// Update takes the representation of a deepOneSpec and updates it. Returns the server's representation of the deepOneSpec, and an error, if there is any.
func (c *deepOneSpecs) Update(deepOneSpec *innsmouth.DeepOneSpec) (result *innsmouth.DeepOneSpec, err error) {
	result = &innsmouth.DeepOneSpec{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("deeponespecs").
		Name(deepOneSpec.Name).
		Body(deepOneSpec).
		Do().
		Into(result)
	return
}

// Delete takes name of the deepOneSpec and deletes it. Returns an error if one occurs.
func (c *deepOneSpecs) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("deeponespecs").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *deepOneSpecs) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("deeponespecs").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the deepOneSpec, and returns the corresponding deepOneSpec object, and an error if there is any.
func (c *deepOneSpecs) Get(name string, options v1.GetOptions) (result *innsmouth.DeepOneSpec, err error) {
	result = &innsmouth.DeepOneSpec{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("deeponespecs").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of DeepOneSpecs that match those selectors.
func (c *deepOneSpecs) List(opts v1.ListOptions) (result *innsmouth.DeepOneSpecList, err error) {
	result = &innsmouth.DeepOneSpecList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("deeponespecs").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested deepOneSpecs.
func (c *deepOneSpecs) Watch(opts v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("deeponespecs").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched deepOneSpec.
func (c *deepOneSpecs) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *innsmouth.DeepOneSpec, err error) {
	result = &innsmouth.DeepOneSpec{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("deeponespecs").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
