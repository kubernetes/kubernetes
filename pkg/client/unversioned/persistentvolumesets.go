/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

type PersistentVolumeSetsInterface interface {
	PersistentVolumeSets() PersistentVolumeSetInterface
}

// PersistentVolumeSetInterface has methods to work with PersistentVolumeSet resources.
type PersistentVolumeSetInterface interface {
	List(label labels.Selector, field fields.Selector) (*api.PersistentVolumeSetList, error)
	Get(name string) (*api.PersistentVolumeSet, error)
	Create(volume *api.PersistentVolumeSet) (*api.PersistentVolumeSet, error)
	Update(volume *api.PersistentVolumeSet) (*api.PersistentVolumeSet, error)
	UpdateStatus(persistentVolumeSet *api.PersistentVolumeSet) (*api.PersistentVolumeSet, error)
	Delete(name string) error
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
}

// persistentVolumeSets implements PersistentVolumeSetsInterface
type persistentVolumeSets struct {
	client *Client
}

func newPersistentVolumeSets(c *Client) *persistentVolumeSets {
	return &persistentVolumeSets{c}
}

func (c *persistentVolumeSets) List(label labels.Selector, field fields.Selector) (result *api.PersistentVolumeSetList, err error) {
	result = &api.PersistentVolumeSetList{}
	err = c.client.Get().
		Resource("persistentVolumeSets").
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Do().
		Into(result)

	return result, err
}

func (c *persistentVolumeSets) Get(name string) (result *api.PersistentVolumeSet, err error) {
	result = &api.PersistentVolumeSet{}
	err = c.client.Get().Resource("persistentVolumeSets").Name(name).Do().Into(result)
	return
}

func (c *persistentVolumeSets) Create(volume *api.PersistentVolumeSet) (result *api.PersistentVolumeSet, err error) {
	result = &api.PersistentVolumeSet{}
	err = c.client.Post().Resource("persistentVolumeSets").Body(volume).Do().Into(result)
	return
}

func (c *persistentVolumeSets) Update(volume *api.PersistentVolumeSet) (result *api.PersistentVolumeSet, err error) {
	result = &api.PersistentVolumeSet{}
	if len(volume.ResourceVersion) == 0 {
		err = fmt.Errorf("invalid update object, missing resource version: %v", volume)
		return
	}
	err = c.client.Put().Resource("persistentVolumeSets").Name(volume.Name).Body(volume).Do().Into(result)
	return
}

func (c *persistentVolumeSets) UpdateStatus(volume *api.PersistentVolumeSet) (result *api.PersistentVolumeSet, err error) {
	result = &api.PersistentVolumeSet{}
	err = c.client.Put().Resource("persistentVolumeSets").Name(volume.Name).SubResource("status").Body(volume).Do().Into(result)
	return
}

func (c *persistentVolumeSets) Delete(name string) error {
	return c.client.Delete().Resource("persistentVolumeSets").Name(name).Do().Error()
}

func (c *persistentVolumeSets) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Resource("persistentVolumeSets").
		Param("resourceVersion", resourceVersion).
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Watch()
}
