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

package internalversion

import (
	api "k8s.io/kubernetes/pkg/api"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	watch "k8s.io/kubernetes/pkg/watch"
)

// PersistentVolumesGetter has a method to return a PersistentVolumeInterface.
// A group's client should implement this interface.
type PersistentVolumesGetter interface {
	PersistentVolumes() PersistentVolumeInterface
}

// PersistentVolumeInterface has methods to work with PersistentVolume resources.
type PersistentVolumeInterface interface {
	Create(*api.PersistentVolume) (*api.PersistentVolume, error)
	Update(*api.PersistentVolume) (*api.PersistentVolume, error)
	UpdateStatus(*api.PersistentVolume) (*api.PersistentVolume, error)
	Delete(name string, options *api.DeleteOptions) error
	DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error
	Get(name string) (*api.PersistentVolume, error)
	List(opts api.ListOptions) (*api.PersistentVolumeList, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *api.PersistentVolume, err error)
	PersistentVolumeExpansion
}

// persistentVolumes implements PersistentVolumeInterface
type persistentVolumes struct {
	client restclient.Interface
}

// newPersistentVolumes returns a PersistentVolumes
func newPersistentVolumes(c *CoreClient) *persistentVolumes {
	return &persistentVolumes{
		client: c.RESTClient(),
	}
}

// Create takes the representation of a persistentVolume and creates it.  Returns the server's representation of the persistentVolume, and an error, if there is any.
func (c *persistentVolumes) Create(persistentVolume *api.PersistentVolume) (result *api.PersistentVolume, err error) {
	result = &api.PersistentVolume{}
	err = c.client.Post().
		Resource("persistentvolumes").
		Body(persistentVolume).
		Do().
		Into(result)
	return
}

// Update takes the representation of a persistentVolume and updates it. Returns the server's representation of the persistentVolume, and an error, if there is any.
func (c *persistentVolumes) Update(persistentVolume *api.PersistentVolume) (result *api.PersistentVolume, err error) {
	result = &api.PersistentVolume{}
	err = c.client.Put().
		Resource("persistentvolumes").
		Name(persistentVolume.Name).
		Body(persistentVolume).
		Do().
		Into(result)
	return
}

func (c *persistentVolumes) UpdateStatus(persistentVolume *api.PersistentVolume) (result *api.PersistentVolume, err error) {
	result = &api.PersistentVolume{}
	err = c.client.Put().
		Resource("persistentvolumes").
		Name(persistentVolume.Name).
		SubResource("status").
		Body(persistentVolume).
		Do().
		Into(result)
	return
}

// Delete takes name of the persistentVolume and deletes it. Returns an error if one occurs.
func (c *persistentVolumes) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().
		Resource("persistentvolumes").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *persistentVolumes) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	return c.client.Delete().
		Resource("persistentvolumes").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the persistentVolume, and returns the corresponding persistentVolume object, and an error if there is any.
func (c *persistentVolumes) Get(name string) (result *api.PersistentVolume, err error) {
	result = &api.PersistentVolume{}
	err = c.client.Get().
		Resource("persistentvolumes").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of PersistentVolumes that match those selectors.
func (c *persistentVolumes) List(opts api.ListOptions) (result *api.PersistentVolumeList, err error) {
	result = &api.PersistentVolumeList{}
	err = c.client.Get().
		Resource("persistentvolumes").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested persistentVolumes.
func (c *persistentVolumes) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Resource("persistentvolumes").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched persistentVolume.
func (c *persistentVolumes) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *api.PersistentVolume, err error) {
	result = &api.PersistentVolume{}
	err = c.client.Patch(pt).
		Resource("persistentvolumes").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
