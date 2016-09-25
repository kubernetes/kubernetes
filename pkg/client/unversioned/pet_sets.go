/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/watch"
)

// PetSetNamespacer has methods to work with PetSet resources in a namespace
type PetSetNamespacer interface {
	PetSets(namespace string) PetSetInterface
}

// PetSetInterface exposes methods to work on PetSet resources.
type PetSetInterface interface {
	List(opts api.ListOptions) (*apps.PetSetList, error)
	Get(name string) (*apps.PetSet, error)
	Create(petSet *apps.PetSet) (*apps.PetSet, error)
	Update(petSet *apps.PetSet) (*apps.PetSet, error)
	Delete(name string, options *api.DeleteOptions) error
	Watch(opts api.ListOptions) (watch.Interface, error)
	UpdateStatus(petSet *apps.PetSet) (*apps.PetSet, error)
}

// petSet implements PetSetNamespacer interface
type petSet struct {
	r  *AppsClient
	ns string
}

// newPetSet returns a petSet
func newPetSet(c *AppsClient, namespace string) *petSet {
	return &petSet{c, namespace}
}

// List returns a list of petSet that match the label and field selectors.
func (c *petSet) List(opts api.ListOptions) (result *apps.PetSetList, err error) {
	result = &apps.PetSetList{}
	err = c.r.Get().Namespace(c.ns).Resource("petsets").VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return
}

// Get returns information about a particular petSet.
func (c *petSet) Get(name string) (result *apps.PetSet, err error) {
	result = &apps.PetSet{}
	err = c.r.Get().Namespace(c.ns).Resource("petsets").Name(name).Do().Into(result)
	return
}

// Create creates a new petSet.
func (c *petSet) Create(petSet *apps.PetSet) (result *apps.PetSet, err error) {
	result = &apps.PetSet{}
	err = c.r.Post().Namespace(c.ns).Resource("petsets").Body(petSet).Do().Into(result)
	return
}

// Update updates an existing petSet.
func (c *petSet) Update(petSet *apps.PetSet) (result *apps.PetSet, err error) {
	result = &apps.PetSet{}
	err = c.r.Put().Namespace(c.ns).Resource("petsets").Name(petSet.Name).Body(petSet).Do().Into(result)
	return
}

// Delete deletes a petSet, returns error if one occurs.
func (c *petSet) Delete(name string, options *api.DeleteOptions) (err error) {
	return c.r.Delete().Namespace(c.ns).Resource("petsets").Name(name).Body(options).Do().Error()
}

// Watch returns a watch.Interface that watches the requested petSet.
func (c *petSet) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.r.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("petsets").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// UpdateStatus takes the name of the petSet and the new status.  Returns the server's representation of the petSet, and an error, if it occurs.
func (c *petSet) UpdateStatus(petSet *apps.PetSet) (result *apps.PetSet, err error) {
	result = &apps.PetSet{}
	err = c.r.Put().Namespace(c.ns).Resource("petsets").Name(petSet.Name).SubResource("status").Body(petSet).Do().Into(result)
	return
}
