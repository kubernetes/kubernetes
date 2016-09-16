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

package fake

import (
	api "k8s.io/kubernetes/pkg/api"
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakePersistentVolumes implements PersistentVolumeInterface
type FakePersistentVolumes struct {
	Fake *FakeCore
}

var persistentvolumesResource = unversioned.GroupVersionResource{Group: "", Version: "", Resource: "persistentvolumes"}

func (c *FakePersistentVolumes) Create(persistentVolume *api.PersistentVolume) (result *api.PersistentVolume, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootCreateAction(persistentvolumesResource, persistentVolume), &api.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolume), err
}

func (c *FakePersistentVolumes) Update(persistentVolume *api.PersistentVolume) (result *api.PersistentVolume, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateAction(persistentvolumesResource, persistentVolume), &api.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolume), err
}

func (c *FakePersistentVolumes) UpdateStatus(persistentVolume *api.PersistentVolume) (*api.PersistentVolume, error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateSubresourceAction(persistentvolumesResource, "status", persistentVolume), &api.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolume), err
}

func (c *FakePersistentVolumes) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewRootDeleteAction(persistentvolumesResource, name), &api.PersistentVolume{})
	return err
}

func (c *FakePersistentVolumes) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewRootDeleteCollectionAction(persistentvolumesResource, listOptions)

	_, err := c.Fake.Invokes(action, &api.PersistentVolumeList{})
	return err
}

func (c *FakePersistentVolumes) Get(name string) (result *api.PersistentVolume, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootGetAction(persistentvolumesResource, name), &api.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolume), err
}

func (c *FakePersistentVolumes) List(opts api.ListOptions) (result *api.PersistentVolumeList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootListAction(persistentvolumesResource, opts), &api.PersistentVolumeList{})
	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &api.PersistentVolumeList{}
	for _, item := range obj.(*api.PersistentVolumeList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested persistentVolumes.
func (c *FakePersistentVolumes) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewRootWatchAction(persistentvolumesResource, opts))
}

// Patch applies the patch and returns the patched persistentVolume.
func (c *FakePersistentVolumes) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *api.PersistentVolume, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootPatchSubresourceAction(persistentvolumesResource, name, data, subresources...), &api.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolume), err
}
