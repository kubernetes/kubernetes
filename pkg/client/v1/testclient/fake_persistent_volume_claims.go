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
	v1api "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

type FakePersistentVolumeClaims struct {
	Fake      *Fake
	Namespace string
}

func (c *FakePersistentVolumeClaims) List(label labels.Selector, field fields.Selector) (*v1api.PersistentVolumeClaimList, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "list-persistentVolumeClaims"}, &v1api.PersistentVolumeClaimList{})
	return obj.(*v1api.PersistentVolumeClaimList), err
}

func (c *FakePersistentVolumeClaims) Get(name string) (*v1api.PersistentVolumeClaim, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "get-persistentVolumeClaims", Value: name}, &v1api.PersistentVolumeClaim{})
	return obj.(*v1api.PersistentVolumeClaim), err
}

func (c *FakePersistentVolumeClaims) Delete(name string) error {
	_, err := c.Fake.Invokes(FakeAction{Action: "delete-persistentVolumeClaims", Value: name}, &v1api.PersistentVolumeClaim{})
	return err
}

func (c *FakePersistentVolumeClaims) Create(claim *v1api.PersistentVolumeClaim) (*v1api.PersistentVolumeClaim, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "create-persistentVolumeClaims"}, &v1api.PersistentVolumeClaim{})
	return obj.(*v1api.PersistentVolumeClaim), err
}

func (c *FakePersistentVolumeClaims) Update(claim *v1api.PersistentVolumeClaim) (*v1api.PersistentVolumeClaim, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "update-persistentVolumeClaims", Value: claim.Name}, &v1api.PersistentVolumeClaim{})
	return obj.(*v1api.PersistentVolumeClaim), err
}

func (c *FakePersistentVolumeClaims) UpdateStatus(claim *v1api.PersistentVolumeClaim) (*v1api.PersistentVolumeClaim, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "update-status-persistentVolumeClaims", Value: claim}, &v1api.PersistentVolumeClaim{})
	return obj.(*v1api.PersistentVolumeClaim), err
}

func (c *FakePersistentVolumeClaims) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "watch-persistentVolumeClaims", Value: resourceVersion})
	return c.Fake.Watch, c.Fake.Err
}
