/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package client

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// PersistentVolumeClaimsNamespacer has methods to work with PersistentVolumeClaim resources in a namespace
type PersistentVolumeClaimsNamespacer interface {
	PersistentVolumeClaims(namespace string) PersistentVolumeClaimInterface
}

// PersistentVolumeClaimInterface has methods to work with PersistentVolumeClaim resources.
type PersistentVolumeClaimInterface interface {
	List(label labels.Selector, field fields.Selector) (*api.PersistentVolumeClaimList, error)
	Get(name string) (*api.PersistentVolumeClaim, error)
	Create(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error)
	Update(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error)
	UpdateStatus(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error)
	Delete(name string) error
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
}

// persistentVolumeClaims implements PersistentVolumeClaimsNamespacer interface
type persistentVolumeClaims struct {
	client    *Client
	namespace string
}

// newPersistentVolumeClaims returns a PodsClient
func newPersistentVolumeClaims(c *Client, namespace string) *persistentVolumeClaims {
	return &persistentVolumeClaims{c, namespace}
}

func (c *persistentVolumeClaims) List(label labels.Selector, field fields.Selector) (result *api.PersistentVolumeClaimList, err error) {
	result = &api.PersistentVolumeClaimList{}

	err = c.client.Get().
		Namespace(c.namespace).
		Resource("persistentVolumeClaims").
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Do().
		Into(result)

	return result, err
}

func (c *persistentVolumeClaims) Get(name string) (result *api.PersistentVolumeClaim, err error) {
	result = &api.PersistentVolumeClaim{}
	err = c.client.Get().Namespace(c.namespace).Resource("persistentVolumeClaims").Name(name).Do().Into(result)
	return
}

func (c *persistentVolumeClaims) Create(claim *api.PersistentVolumeClaim) (result *api.PersistentVolumeClaim, err error) {
	result = &api.PersistentVolumeClaim{}
	err = c.client.Post().Namespace(c.namespace).Resource("persistentVolumeClaims").Body(claim).Do().Into(result)
	return
}

func (c *persistentVolumeClaims) Update(claim *api.PersistentVolumeClaim) (result *api.PersistentVolumeClaim, err error) {
	result = &api.PersistentVolumeClaim{}
	if len(claim.ResourceVersion) == 0 {
		err = fmt.Errorf("invalid update object, missing resource version: %v", claim)
		return
	}
	err = c.client.Put().Namespace(c.namespace).Resource("persistentVolumeClaims").Name(claim.Name).Body(claim).Do().Into(result)
	return
}

func (c *persistentVolumeClaims) UpdateStatus(claim *api.PersistentVolumeClaim) (result *api.PersistentVolumeClaim, err error) {
	result = &api.PersistentVolumeClaim{}
	err = c.client.Put().Namespace(c.namespace).Resource("persistentVolumeClaims").Name(claim.Name).SubResource("status").Body(claim).Do().Into(result)
	return
}

func (c *persistentVolumeClaims) Delete(name string) error {
	return c.client.Delete().Namespace(c.namespace).Resource("persistentVolumeClaims").Name(name).Do().Error()
}

func (c *persistentVolumeClaims) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.namespace).
		Resource("persistentVolumeClaims").
		Param("resourceVersion", resourceVersion).
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Watch()
}
