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
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

type FakePersistentVolumes struct {
	Fake *Fake
}

func (c *FakePersistentVolumes) Get(name string) (*api.PersistentVolume, error) {
	obj, err := c.Fake.Invokes(NewRootGetAction("persistentvolumes", name), &api.PersistentVolume{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.PersistentVolume), err
}

func (c *FakePersistentVolumes) List(label labels.Selector, field fields.Selector) (*api.PersistentVolumeList, error) {
	obj, err := c.Fake.Invokes(NewRootListAction("persistentvolumes", label, field), &api.PersistentVolumeList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.PersistentVolumeList), err
}

func (c *FakePersistentVolumes) Create(pv *api.PersistentVolume) (*api.PersistentVolume, error) {
	obj, err := c.Fake.Invokes(NewRootCreateAction("persistentvolumes", pv), pv)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.PersistentVolume), err
}

func (c *FakePersistentVolumes) Update(pv *api.PersistentVolume) (*api.PersistentVolume, error) {
	obj, err := c.Fake.Invokes(NewRootUpdateAction("persistentvolumes", pv), pv)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.PersistentVolume), err
}

func (c *FakePersistentVolumes) Delete(name string) error {
	_, err := c.Fake.Invokes(NewRootDeleteAction("persistentvolumes", name), &api.PersistentVolume{})
	return err
}

func (c *FakePersistentVolumes) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewRootWatchAction("persistentvolumes", label, field, resourceVersion))
}

func (c *FakePersistentVolumes) UpdateStatus(pv *api.PersistentVolume) (*api.PersistentVolume, error) {
	action := UpdateActionImpl{}
	action.Verb = "update"
	action.Resource = "persistentvolumes"
	action.Subresource = "status"
	action.Object = pv

	obj, err := c.Fake.Invokes(action, pv)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.PersistentVolume), err
}
