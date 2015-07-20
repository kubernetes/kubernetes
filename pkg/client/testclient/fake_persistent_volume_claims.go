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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

type FakePersistentVolumeClaims struct {
	Fake      *Fake
	Namespace string
}

func (c *FakePersistentVolumeClaims) List(label labels.Selector, field fields.Selector) (*api.PersistentVolumeClaimList, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "list-persistentVolumeClaims"}, &api.PersistentVolumeClaimList{})
	return obj.(*api.PersistentVolumeClaimList), err
}

func (c *FakePersistentVolumeClaims) Get(name string) (*api.PersistentVolumeClaim, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "get-persistentVolumeClaims", Value: name}, &api.PersistentVolumeClaim{})
	return obj.(*api.PersistentVolumeClaim), err
}

func (c *FakePersistentVolumeClaims) Delete(name string) error {
	_, err := c.Fake.Invokes(FakeAction{Action: "delete-persistentVolumeClaims", Value: name}, &api.PersistentVolumeClaim{})
	return err
}

func (c *FakePersistentVolumeClaims) Create(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "create-persistentVolumeClaims"}, &api.PersistentVolumeClaim{})
	return obj.(*api.PersistentVolumeClaim), err
}

func (c *FakePersistentVolumeClaims) Update(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "update-persistentVolumeClaims", Value: claim.Name}, &api.PersistentVolumeClaim{})
	return obj.(*api.PersistentVolumeClaim), err
}

func (c *FakePersistentVolumeClaims) UpdateStatus(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "update-status-persistentVolumeClaims", Value: claim}, &api.PersistentVolumeClaim{})
	return obj.(*api.PersistentVolumeClaim), err
}

func (c *FakePersistentVolumeClaims) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Lock.Lock()
	defer c.Fake.Lock.Unlock()

	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "watch-persistentVolumeClaims", Value: resourceVersion})
	return c.Fake.Watch, c.Fake.Err
}
