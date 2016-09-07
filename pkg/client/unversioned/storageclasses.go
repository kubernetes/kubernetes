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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/watch"
)

type StorageClassesInterface interface {
	StorageClasses() StorageClassInterface
}

// StorageClassInterface has methods to work with StorageClass resources.
type StorageClassInterface interface {
	List(opts api.ListOptions) (*storage.StorageClassList, error)
	Get(name string) (*storage.StorageClass, error)
	Create(storageClass *storage.StorageClass) (*storage.StorageClass, error)
	Update(storageClass *storage.StorageClass) (*storage.StorageClass, error)
	Delete(name string) error
	Watch(opts api.ListOptions) (watch.Interface, error)
}

// storageClasses implements StorageClassInterface
type storageClasses struct {
	client *StorageClient
}

func newStorageClasses(c *StorageClient) *storageClasses {
	return &storageClasses{c}
}

func (c *storageClasses) List(opts api.ListOptions) (result *storage.StorageClassList, err error) {
	result = &storage.StorageClassList{}
	err = c.client.Get().
		Resource("storageclasses").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)

	return result, err
}

func (c *storageClasses) Get(name string) (result *storage.StorageClass, err error) {
	result = &storage.StorageClass{}
	err = c.client.Get().Resource("storageClasses").Name(name).Do().Into(result)
	return
}

func (c *storageClasses) Create(storageClass *storage.StorageClass) (result *storage.StorageClass, err error) {
	result = &storage.StorageClass{}
	err = c.client.Post().Resource("storageClasses").Body(storageClass).Do().Into(result)
	return
}

func (c *storageClasses) Update(storageClass *storage.StorageClass) (result *storage.StorageClass, err error) {
	result = &storage.StorageClass{}
	err = c.client.Put().Resource("storageClasses").Name(storageClass.Name).Body(storageClass).Do().Into(result)
	return
}

func (c *storageClasses) Delete(name string) error {
	return c.client.Delete().Resource("storageClasses").Name(name).Do().Error()
}

func (c *storageClasses) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Resource("storageClasses").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}
