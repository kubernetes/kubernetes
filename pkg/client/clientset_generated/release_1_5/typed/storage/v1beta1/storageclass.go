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

package v1beta1

import (
	api "k8s.io/kubernetes/pkg/api"
	v1 "k8s.io/kubernetes/pkg/api/v1"
	v1beta1 "k8s.io/kubernetes/pkg/apis/storage/v1beta1"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	watch "k8s.io/kubernetes/pkg/watch"
)

// StorageClassesGetter has a method to return a StorageClassInterface.
// A group's client should implement this interface.
type StorageClassesGetter interface {
	StorageClasses() StorageClassInterface
}

// StorageClassInterface has methods to work with StorageClass resources.
type StorageClassInterface interface {
	Create(*v1beta1.StorageClass) (*v1beta1.StorageClass, error)
	Update(*v1beta1.StorageClass) (*v1beta1.StorageClass, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string) (*v1beta1.StorageClass, error)
	List(opts v1.ListOptions) (*v1beta1.StorageClassList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1beta1.StorageClass, err error)
	StorageClassExpansion
}

// storageClasses implements StorageClassInterface
type storageClasses struct {
	client restclient.Interface
}

// newStorageClasses returns a StorageClasses
func newStorageClasses(c *StorageV1beta1Client) *storageClasses {
	return &storageClasses{
		client: c.RESTClient(),
	}
}

// Create takes the representation of a storageClass and creates it.  Returns the server's representation of the storageClass, and an error, if there is any.
func (c *storageClasses) Create(storageClass *v1beta1.StorageClass) (result *v1beta1.StorageClass, err error) {
	result = &v1beta1.StorageClass{}
	err = c.client.Post().
		Resource("storageclasses").
		Body(storageClass).
		Do().
		Into(result)
	return
}

// Update takes the representation of a storageClass and updates it. Returns the server's representation of the storageClass, and an error, if there is any.
func (c *storageClasses) Update(storageClass *v1beta1.StorageClass) (result *v1beta1.StorageClass, err error) {
	result = &v1beta1.StorageClass{}
	err = c.client.Put().
		Resource("storageclasses").
		Name(storageClass.Name).
		Body(storageClass).
		Do().
		Into(result)
	return
}

// Delete takes name of the storageClass and deletes it. Returns an error if one occurs.
func (c *storageClasses) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Resource("storageclasses").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *storageClasses) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Resource("storageclasses").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the storageClass, and returns the corresponding storageClass object, and an error if there is any.
func (c *storageClasses) Get(name string) (result *v1beta1.StorageClass, err error) {
	result = &v1beta1.StorageClass{}
	err = c.client.Get().
		Resource("storageclasses").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of StorageClasses that match those selectors.
func (c *storageClasses) List(opts v1.ListOptions) (result *v1beta1.StorageClassList, err error) {
	result = &v1beta1.StorageClassList{}
	err = c.client.Get().
		Resource("storageclasses").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested storageClasses.
func (c *storageClasses) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Resource("storageclasses").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched storageClass.
func (c *storageClasses) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1beta1.StorageClass, err error) {
	result = &v1beta1.StorageClass{}
	err = c.client.Patch(pt).
		Resource("storageclasses").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
