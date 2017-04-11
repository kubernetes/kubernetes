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

// DeepOnesGetter has a method to return a DeepOneInterface.
// A group's client should implement this interface.
type DeepOnesGetter interface {
	DeepOnes(namespace string) DeepOneInterface
}

// DeepOneInterface has methods to work with DeepOne resources.
type DeepOneInterface interface {
	Create(*innsmouth.DeepOne) (*innsmouth.DeepOne, error)
	Update(*innsmouth.DeepOne) (*innsmouth.DeepOne, error)
	UpdateStatus(*innsmouth.DeepOne) (*innsmouth.DeepOne, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*innsmouth.DeepOne, error)
	List(opts v1.ListOptions) (*innsmouth.DeepOneList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *innsmouth.DeepOne, err error)
	DeepOneExpansion
}

// deepOnes implements DeepOneInterface
type deepOnes struct {
	client rest.Interface
	ns     string
}

// newDeepOnes returns a DeepOnes
func newDeepOnes(c *InnsmouthClient, namespace string) *deepOnes {
	return &deepOnes{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a deepOne and creates it.  Returns the server's representation of the deepOne, and an error, if there is any.
func (c *deepOnes) Create(deepOne *innsmouth.DeepOne) (result *innsmouth.DeepOne, err error) {
	result = &innsmouth.DeepOne{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("deepones").
		Body(deepOne).
		Do().
		Into(result)
	return
}

// Update takes the representation of a deepOne and updates it. Returns the server's representation of the deepOne, and an error, if there is any.
func (c *deepOnes) Update(deepOne *innsmouth.DeepOne) (result *innsmouth.DeepOne, err error) {
	result = &innsmouth.DeepOne{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("deepones").
		Name(deepOne.Name).
		Body(deepOne).
		Do().
		Into(result)
	return
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclientstatus=false comment above the type to avoid generating UpdateStatus().

func (c *deepOnes) UpdateStatus(deepOne *innsmouth.DeepOne) (result *innsmouth.DeepOne, err error) {
	result = &innsmouth.DeepOne{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("deepones").
		Name(deepOne.Name).
		SubResource("status").
		Body(deepOne).
		Do().
		Into(result)
	return
}

// Delete takes name of the deepOne and deletes it. Returns an error if one occurs.
func (c *deepOnes) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("deepones").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *deepOnes) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("deepones").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the deepOne, and returns the corresponding deepOne object, and an error if there is any.
func (c *deepOnes) Get(name string, options v1.GetOptions) (result *innsmouth.DeepOne, err error) {
	result = &innsmouth.DeepOne{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("deepones").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of DeepOnes that match those selectors.
func (c *deepOnes) List(opts v1.ListOptions) (result *innsmouth.DeepOneList, err error) {
	result = &innsmouth.DeepOneList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("deepones").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested deepOnes.
func (c *deepOnes) Watch(opts v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("deepones").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched deepOne.
func (c *deepOnes) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *innsmouth.DeepOne, err error) {
	result = &innsmouth.DeepOne{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("deepones").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
