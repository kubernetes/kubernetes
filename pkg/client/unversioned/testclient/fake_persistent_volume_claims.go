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

package testclient

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/watch"
)

type FakePersistentVolumeClaims struct {
	Fake      *Fake
	Namespace string
}

func (c *FakePersistentVolumeClaims) Get(name string) (*api.PersistentVolumeClaim, error) {
	obj, err := c.Fake.Invokes(NewGetAction("persistentvolumeclaims", c.Namespace, name), &api.PersistentVolumeClaim{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.PersistentVolumeClaim), err
}

func (c *FakePersistentVolumeClaims) List(opts api.ListOptions) (*api.PersistentVolumeClaimList, error) {
	obj, err := c.Fake.Invokes(NewListAction("persistentvolumeclaims", c.Namespace, opts), &api.PersistentVolumeClaimList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.PersistentVolumeClaimList), err
}

func (c *FakePersistentVolumeClaims) Create(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("persistentvolumeclaims", c.Namespace, claim), claim)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.PersistentVolumeClaim), err
}

func (c *FakePersistentVolumeClaims) Update(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("persistentvolumeclaims", c.Namespace, claim), claim)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.PersistentVolumeClaim), err
}

func (c *FakePersistentVolumeClaims) Delete(name string) error {
	_, err := c.Fake.Invokes(NewDeleteAction("persistentvolumeclaims", c.Namespace, name), &api.PersistentVolumeClaim{})
	return err
}

func (c *FakePersistentVolumeClaims) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("persistentvolumeclaims", c.Namespace, opts))
}

func (c *FakePersistentVolumeClaims) UpdateStatus(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	obj, err := c.Fake.Invokes(NewUpdateSubresourceAction("persistentvolumeclaims", "status", c.Namespace, claim), claim)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.PersistentVolumeClaim), err
}
