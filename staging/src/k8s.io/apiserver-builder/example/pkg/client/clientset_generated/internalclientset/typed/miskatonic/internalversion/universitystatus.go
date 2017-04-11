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

// UniversityStatusesGetter has a method to return a UniversityStatusInterface.
// A group's client should implement this interface.
type UniversityStatusesGetter interface {
	UniversityStatuses(namespace string) UniversityStatusInterface
}

// UniversityStatusInterface has methods to work with UniversityStatus resources.
type UniversityStatusInterface interface {
	Create(*miskatonic.UniversityStatus) (*miskatonic.UniversityStatus, error)
	Update(*miskatonic.UniversityStatus) (*miskatonic.UniversityStatus, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*miskatonic.UniversityStatus, error)
	List(opts v1.ListOptions) (*miskatonic.UniversityStatusList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *miskatonic.UniversityStatus, err error)
	UniversityStatusExpansion
}

// universityStatuses implements UniversityStatusInterface
type universityStatuses struct {
	client rest.Interface
	ns     string
}

// newUniversityStatuses returns a UniversityStatuses
func newUniversityStatuses(c *MiskatonicClient, namespace string) *universityStatuses {
	return &universityStatuses{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a universityStatus and creates it.  Returns the server's representation of the universityStatus, and an error, if there is any.
func (c *universityStatuses) Create(universityStatus *miskatonic.UniversityStatus) (result *miskatonic.UniversityStatus, err error) {
	result = &miskatonic.UniversityStatus{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("universitystatuses").
		Body(universityStatus).
		Do().
		Into(result)
	return
}

// Update takes the representation of a universityStatus and updates it. Returns the server's representation of the universityStatus, and an error, if there is any.
func (c *universityStatuses) Update(universityStatus *miskatonic.UniversityStatus) (result *miskatonic.UniversityStatus, err error) {
	result = &miskatonic.UniversityStatus{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("universitystatuses").
		Name(universityStatus.Name).
		Body(universityStatus).
		Do().
		Into(result)
	return
}

// Delete takes name of the universityStatus and deletes it. Returns an error if one occurs.
func (c *universityStatuses) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("universitystatuses").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *universityStatuses) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("universitystatuses").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the universityStatus, and returns the corresponding universityStatus object, and an error if there is any.
func (c *universityStatuses) Get(name string, options v1.GetOptions) (result *miskatonic.UniversityStatus, err error) {
	result = &miskatonic.UniversityStatus{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("universitystatuses").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of UniversityStatuses that match those selectors.
func (c *universityStatuses) List(opts v1.ListOptions) (result *miskatonic.UniversityStatusList, err error) {
	result = &miskatonic.UniversityStatusList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("universitystatuses").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested universityStatuses.
func (c *universityStatuses) Watch(opts v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("universitystatuses").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched universityStatus.
func (c *universityStatuses) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *miskatonic.UniversityStatus, err error) {
	result = &miskatonic.UniversityStatus{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("universitystatuses").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
