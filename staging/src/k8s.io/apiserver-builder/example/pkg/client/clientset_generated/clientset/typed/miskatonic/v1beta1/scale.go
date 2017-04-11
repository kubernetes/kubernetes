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

package v1beta1

import (
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	v1beta1 "k8s.io/apiserver-builder/example/pkg/apis/miskatonic/v1beta1"
	scheme "k8s.io/apiserver-builder/example/pkg/client/clientset_generated/clientset/scheme"
	rest "k8s.io/client-go/rest"
)

// ScalesGetter has a method to return a ScaleInterface.
// A group's client should implement this interface.
type ScalesGetter interface {
	Scales(namespace string) ScaleInterface
}

// ScaleInterface has methods to work with Scale resources.
type ScaleInterface interface {
	Create(*v1beta1.Scale) (*v1beta1.Scale, error)
	Update(*v1beta1.Scale) (*v1beta1.Scale, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*v1beta1.Scale, error)
	List(opts v1.ListOptions) (*v1beta1.ScaleList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1beta1.Scale, err error)
	ScaleExpansion
}

// scales implements ScaleInterface
type scales struct {
	client rest.Interface
	ns     string
}

// newScales returns a Scales
func newScales(c *MiskatonicV1beta1Client, namespace string) *scales {
	return &scales{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a scale and creates it.  Returns the server's representation of the scale, and an error, if there is any.
func (c *scales) Create(scale *v1beta1.Scale) (result *v1beta1.Scale, err error) {
	result = &v1beta1.Scale{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("scales").
		Body(scale).
		Do().
		Into(result)
	return
}

// Update takes the representation of a scale and updates it. Returns the server's representation of the scale, and an error, if there is any.
func (c *scales) Update(scale *v1beta1.Scale) (result *v1beta1.Scale, err error) {
	result = &v1beta1.Scale{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("scales").
		Name(scale.Name).
		Body(scale).
		Do().
		Into(result)
	return
}

// Delete takes name of the scale and deletes it. Returns an error if one occurs.
func (c *scales) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("scales").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *scales) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("scales").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the scale, and returns the corresponding scale object, and an error if there is any.
func (c *scales) Get(name string, options v1.GetOptions) (result *v1beta1.Scale, err error) {
	result = &v1beta1.Scale{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("scales").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Scales that match those selectors.
func (c *scales) List(opts v1.ListOptions) (result *v1beta1.ScaleList, err error) {
	result = &v1beta1.ScaleList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("scales").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested scales.
func (c *scales) Watch(opts v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("scales").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched scale.
func (c *scales) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1beta1.Scale, err error) {
	result = &v1beta1.Scale{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("scales").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
