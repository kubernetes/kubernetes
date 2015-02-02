/*
Copyright 2014 Google Inc. All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// FakePods implements PodsInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakePersistentVolumeClaims struct {
	Fake      *Fake
	Namespace string
}

func (c *FakePersistentVolumeClaims) List(selector labels.Selector) (*api.PersistentVolumeClaimList, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "list-persistentVolumeClaims"})
	return api.Scheme.CopyOrDie(&c.Fake.PersistentVolumeClaimList).(*api.PersistentVolumeClaimList), nil
}

func (c *FakePersistentVolumeClaims) Get(name string) (*api.PersistentVolumeClaim, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "get-persistentVolumeClaim", Value: name})
	return &api.PersistentVolumeClaim{ObjectMeta: api.ObjectMeta{Name: name, Namespace: c.Namespace}}, nil
}

func (c *FakePersistentVolumeClaims) Delete(name string) error {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "delete-persistentVolumeClaim", Value: name})
	return nil
}

func (c *FakePersistentVolumeClaims) Create(persistentvolumeclaim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "create-persistentVolumeClaim"})
	return &api.PersistentVolumeClaim{}, nil
}

func (c *FakePersistentVolumeClaims) Update(persistentvolumeclaim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "update-persistentVolumeClaim", Value: persistentvolumeclaim.Name})
	return &api.PersistentVolumeClaim{}, nil
}

func (c *FakePersistentVolumeClaims) Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	return nil, nil
}
