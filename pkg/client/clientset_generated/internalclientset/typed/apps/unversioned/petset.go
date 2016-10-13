/*
Copyright 2016 The Kubernetes Authors.

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
	api "k8s.io/kubernetes/pkg/api"
	apps "k8s.io/kubernetes/pkg/apis/apps"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	watch "k8s.io/kubernetes/pkg/watch"
)

// PetSetsGetter has a method to return a PetSetInterface.
// A group's client should implement this interface.
type PetSetsGetter interface {
	PetSets(namespace string) PetSetInterface
}

// PetSetInterface has methods to work with PetSet resources.
type PetSetInterface interface {
	Create(*apps.PetSet) (*apps.PetSet, error)
	Update(*apps.PetSet) (*apps.PetSet, error)
	UpdateStatus(*apps.PetSet) (*apps.PetSet, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*apps.PetSet, error)
	List(opts api.ListOptions) (*apps.PetSetList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *apps.PetSet, err error)
	PetSetExpansion
}

// petSets implements PetSetInterface
type petSets struct {
	client restclient.Interface
	ns     string
}

// newPetSets returns a PetSets
func newPetSets(c *AppsClient, namespace string) *petSets {
	return &petSets{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a petSet and creates it.  Returns the server's representation of the petSet, and an error, if there is any.
func (c *petSets) Create(petSet *apps.PetSet) (result *apps.PetSet, err error) {
	result = &apps.PetSet{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("petsets").
		Body(petSet).
		Do().
		Into(result)
	return
}

// Update takes the representation of a petSet and updates it. Returns the server's representation of the petSet, and an error, if there is any.
func (c *petSets) Update(petSet *apps.PetSet) (result *apps.PetSet, err error) {
	result = &apps.PetSet{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("petsets").
		Name(petSet.Name).
		Body(petSet).
		Do().
		Into(result)
	return
}

func (c *petSets) UpdateStatus(petSet *apps.PetSet) (result *apps.PetSet, err error) {
	result = &apps.PetSet{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("petsets").
		Name(petSet.Name).
		SubResource("status").
		Body(petSet).
		Do().
		Into(result)
	return
}

// Delete takes name of the petSet and deletes it. Returns an error if one occurs.
func (c *petSets) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("petsets").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *petSets) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("petsets").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the petSet, and returns the corresponding petSet object, and an error if there is any.
func (c *petSets) Get(name string) (result *apps.PetSet, err error) {
	result = &apps.PetSet{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("petsets").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of PetSets that match those selectors.
func (c *petSets) List(opts api.ListOptions) (result *apps.PetSetList, err error) {
	result = &apps.PetSetList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("petsets").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested petSets.
func (c *petSets) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("petsets").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched petSet.
func (c *petSets) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *apps.PetSet, err error) {
	result = &apps.PetSet{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("petsets").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
