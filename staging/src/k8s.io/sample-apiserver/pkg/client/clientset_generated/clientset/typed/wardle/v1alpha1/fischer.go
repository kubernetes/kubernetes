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
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	rest "k8s.io/client-go/rest"
	v1alpha1 "k8s.io/sample-apiserver/pkg/apis/wardle/v1alpha1"
	scheme "k8s.io/sample-apiserver/pkg/client/clientset_generated/clientset/scheme"
)

// FischersGetter has a method to return a FischerInterface.
// A group's client should implement this interface.
type FischersGetter interface {
	Fischers() FischerInterface
}

// FischerInterface has methods to work with Fischer resources.
type FischerInterface interface {
	Create(*v1alpha1.Fischer) (*v1alpha1.Fischer, error)
	Update(*v1alpha1.Fischer) (*v1alpha1.Fischer, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*v1alpha1.Fischer, error)
	List(opts v1.ListOptions) (*v1alpha1.FischerList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1alpha1.Fischer, err error)
	FischerExpansion
}

// fischers implements FischerInterface
type fischers struct {
	client rest.Interface
}

// newFischers returns a Fischers
func newFischers(c *WardleV1alpha1Client) *fischers {
	return &fischers{
		client: c.RESTClient(),
	}
}

// Get takes name of the fischer, and returns the corresponding fischer object, and an error if there is any.
func (c *fischers) Get(name string, options v1.GetOptions) (result *v1alpha1.Fischer, err error) {
	result = &v1alpha1.Fischer{}
	err = c.client.Get().
		Resource("fischers").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Fischers that match those selectors.
func (c *fischers) List(opts v1.ListOptions) (result *v1alpha1.FischerList, err error) {
	result = &v1alpha1.FischerList{}
	err = c.client.Get().
		Resource("fischers").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested fischers.
func (c *fischers) Watch(opts v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Resource("fischers").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}

// Create takes the representation of a fischer and creates it.  Returns the server's representation of the fischer, and an error, if there is any.
func (c *fischers) Create(fischer *v1alpha1.Fischer) (result *v1alpha1.Fischer, err error) {
	result = &v1alpha1.Fischer{}
	err = c.client.Post().
		Resource("fischers").
		Body(fischer).
		Do().
		Into(result)
	return
}

// Update takes the representation of a fischer and updates it. Returns the server's representation of the fischer, and an error, if there is any.
func (c *fischers) Update(fischer *v1alpha1.Fischer) (result *v1alpha1.Fischer, err error) {
	result = &v1alpha1.Fischer{}
	err = c.client.Put().
		Resource("fischers").
		Name(fischer.Name).
		Body(fischer).
		Do().
		Into(result)
	return
}

// Delete takes name of the fischer and deletes it. Returns an error if one occurs.
func (c *fischers) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Resource("fischers").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *fischers) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Resource("fischers").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Patch applies the patch and returns the patched fischer.
func (c *fischers) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1alpha1.Fischer, err error) {
	result = &v1alpha1.Fischer{}
	err = c.client.Patch(pt).
		Resource("fischers").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
