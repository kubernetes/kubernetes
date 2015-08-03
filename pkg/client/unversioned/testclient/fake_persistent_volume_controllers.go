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

package testclient

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

type FakePersistentVolumeSets struct {
	Fake      *Fake
	Namespace string
}

func (c *FakePersistentVolumeSets) List(label labels.Selector, field fields.Selector) (*api.PersistentVolumeSetList, error) {
	obj, err := c.Fake.Invokes(NewRootListAction("persistentvolumesets", label, field), &api.PersistentVolumeSetList{})
	return obj.(*api.PersistentVolumeSetList), err
}

func (c *FakePersistentVolumeSets) Get(name string) (*api.PersistentVolumeSet, error) {
	obj, err := c.Fake.Invokes(NewRootGetAction("persistentvolumesets", name), &api.PersistentVolumeSet{})
	return obj.(*api.PersistentVolumeSet), err
}

func (c *FakePersistentVolumeSets) Delete(name string) error {
	_, err := c.Fake.Invokes(NewRootDeleteAction("persistentvolumesets", name), &api.PersistentVolumeSet{})
	return err
}

func (c *FakePersistentVolumeSets) Create(pv *api.PersistentVolumeSet) (*api.PersistentVolumeSet, error) {
	obj, err := c.Fake.Invokes(NewRootCreateAction("persistentvolumesets", pv), &api.PersistentVolumeSet{})
	return obj.(*api.PersistentVolumeSet), err
}

func (c *FakePersistentVolumeSets) Update(pv *api.PersistentVolumeSet) (*api.PersistentVolumeSet, error) {
	obj, err := c.Fake.Invokes(NewRootUpdateAction("persistentvolumesets", pv), &api.PersistentVolumeSet{})
	return obj.(*api.PersistentVolumeSet), err
}

func (c *FakePersistentVolumeSets) UpdateStatus(pv *api.PersistentVolumeSet) (*api.PersistentVolumeSet, error) {
	action := UpdateActionImpl{}
	action.Verb = "update"
	action.Resource = "persistentvolumesets"
	action.Subresource = "status"
	action.Object = pv

	obj, err := c.Fake.Invokes(action, pv)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.PersistentVolumeSet), err
}

func (c *FakePersistentVolumeSets) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewRootWatchAction("persistentvolumesets", label, field, resourceVersion))
}
