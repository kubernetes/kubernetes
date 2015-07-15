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

type FakePersistentVolumes struct {
	Fake      *Fake
	Namespace string
}

func (c *FakePersistentVolumes) List(label labels.Selector, field fields.Selector) (*v1api.PersistentVolumeList, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "list-persistentVolumes"}, &v1api.PersistentVolumeList{})
	return obj.(*v1api.PersistentVolumeList), err
}

func (c *FakePersistentVolumes) Get(name string) (*v1api.PersistentVolume, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "get-persistentVolumes", Value: name}, &v1api.PersistentVolume{})
	return obj.(*v1api.PersistentVolume), err
}

func (c *FakePersistentVolumes) Delete(name string) error {
	_, err := c.Fake.Invokes(FakeAction{Action: "delete-persistentVolumes", Value: name}, &v1api.PersistentVolume{})
	return err
}

func (c *FakePersistentVolumes) Create(pv *v1api.PersistentVolume) (*v1api.PersistentVolume, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "create-persistentVolumes"}, &v1api.PersistentVolume{})
	return obj.(*v1api.PersistentVolume), err
}

func (c *FakePersistentVolumes) Update(pv *v1api.PersistentVolume) (*v1api.PersistentVolume, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "update-persistentVolumes", Value: pv.Name}, &v1api.PersistentVolume{})
	return obj.(*v1api.PersistentVolume), err
}

func (c *FakePersistentVolumes) UpdateStatus(pv *v1api.PersistentVolume) (*v1api.PersistentVolume, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "update-status-persistentVolumes", Value: pv}, &v1api.PersistentVolume{})
	return obj.(*v1api.PersistentVolume), err
}

func (c *FakePersistentVolumes) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "watch-persistentVolumes", Value: resourceVersion})
	return c.Fake.Watch, c.Fake.Err
}
