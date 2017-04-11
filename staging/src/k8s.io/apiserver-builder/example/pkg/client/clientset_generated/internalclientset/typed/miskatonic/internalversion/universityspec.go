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
	miskatonic "k8s.io/apiserver-builder/example/pkg/apis/miskatonic"
	scheme "k8s.io/apiserver-builder/example/pkg/client/clientset_generated/internalclientset/scheme"
	rest "k8s.io/client-go/rest"
)

// UniversitySpecsGetter has a method to return a UniversitySpecInterface.
// A group's client should implement this interface.
type UniversitySpecsGetter interface {
	UniversitySpecs(namespace string) UniversitySpecInterface
}

// UniversitySpecInterface has methods to work with UniversitySpec resources.
type UniversitySpecInterface interface {
	Create(*miskatonic.UniversitySpec) (*miskatonic.UniversitySpec, error)
	Update(*miskatonic.UniversitySpec) (*miskatonic.UniversitySpec, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*miskatonic.UniversitySpec, error)
	List(opts v1.ListOptions) (*miskatonic.UniversitySpecList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *miskatonic.UniversitySpec, err error)
	UniversitySpecExpansion
}

// universitySpecs implements UniversitySpecInterface
type universitySpecs struct {
	client rest.Interface
	ns     string
}

// newUniversitySpecs returns a UniversitySpecs
func newUniversitySpecs(c *MiskatonicClient, namespace string) *universitySpecs {
	return &universitySpecs{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a universitySpec and creates it.  Returns the server's representation of the universitySpec, and an error, if there is any.
func (c *universitySpecs) Create(universitySpec *miskatonic.UniversitySpec) (result *miskatonic.UniversitySpec, err error) {
	result = &miskatonic.UniversitySpec{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("universityspecs").
		Body(universitySpec).
		Do().
		Into(result)
	return
}

// Update takes the representation of a universitySpec and updates it. Returns the server's representation of the universitySpec, and an error, if there is any.
func (c *universitySpecs) Update(universitySpec *miskatonic.UniversitySpec) (result *miskatonic.UniversitySpec, err error) {
	result = &miskatonic.UniversitySpec{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("universityspecs").
		Name(universitySpec.Name).
		Body(universitySpec).
		Do().
		Into(result)
	return
}

// Delete takes name of the universitySpec and deletes it. Returns an error if one occurs.
func (c *universitySpecs) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("universityspecs").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *universitySpecs) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("universityspecs").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the universitySpec, and returns the corresponding universitySpec object, and an error if there is any.
func (c *universitySpecs) Get(name string, options v1.GetOptions) (result *miskatonic.UniversitySpec, err error) {
	result = &miskatonic.UniversitySpec{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("universityspecs").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of UniversitySpecs that match those selectors.
func (c *universitySpecs) List(opts v1.ListOptions) (result *miskatonic.UniversitySpecList, err error) {
	result = &miskatonic.UniversitySpecList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("universityspecs").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested universitySpecs.
func (c *universitySpecs) Watch(opts v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("universityspecs").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched universitySpec.
func (c *universitySpecs) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *miskatonic.UniversitySpec, err error) {
	result = &miskatonic.UniversitySpec{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("universityspecs").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
