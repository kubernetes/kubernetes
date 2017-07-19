/*
Copyright 2017 The Kubernetes Authors.

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
	core_v1 "k8s.io/api/core/v1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	testing "k8s.io/client-go/testing"
)

// FakePersistentVolumes implements PersistentVolumeInterface
type FakePersistentVolumes struct {
	Fake *FakeCoreV1
}

var persistentvolumesResource = schema.GroupVersionResource{Group: "", Version: "v1", Resource: "persistentvolumes"}

var persistentvolumesKind = schema.GroupVersionKind{Group: "", Version: "v1", Kind: "PersistentVolume"}

func (c *FakePersistentVolumes) Get(name string, options v1.GetOptions) (result *core_v1.PersistentVolume, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(persistentvolumesResource, name), &core_v1.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*core_v1.PersistentVolume), err
}

func (c *FakePersistentVolumes) List(opts v1.ListOptions) (result *core_v1.PersistentVolumeList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(persistentvolumesResource, persistentvolumesKind, opts), &core_v1.PersistentVolumeList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &core_v1.PersistentVolumeList{}
	for _, item := range obj.(*core_v1.PersistentVolumeList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested persistentVolumes.
func (c *FakePersistentVolumes) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(persistentvolumesResource, opts))
}

func (c *FakePersistentVolumes) Create(persistentVolume *core_v1.PersistentVolume) (result *core_v1.PersistentVolume, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(persistentvolumesResource, persistentVolume), &core_v1.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*core_v1.PersistentVolume), err
}

func (c *FakePersistentVolumes) Update(persistentVolume *core_v1.PersistentVolume) (result *core_v1.PersistentVolume, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(persistentvolumesResource, persistentVolume), &core_v1.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*core_v1.PersistentVolume), err
}

func (c *FakePersistentVolumes) UpdateStatus(persistentVolume *core_v1.PersistentVolume) (*core_v1.PersistentVolume, error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateSubresourceAction(persistentvolumesResource, "status", persistentVolume), &core_v1.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*core_v1.PersistentVolume), err
}

func (c *FakePersistentVolumes) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(persistentvolumesResource, name), &core_v1.PersistentVolume{})
	return err
}

func (c *FakePersistentVolumes) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(persistentvolumesResource, listOptions)

	_, err := c.Fake.Invokes(action, &core_v1.PersistentVolumeList{})
	return err
}

// Patch applies the patch and returns the patched persistentVolume.
func (c *FakePersistentVolumes) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *core_v1.PersistentVolume, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(persistentvolumesResource, name, data, subresources...), &core_v1.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*core_v1.PersistentVolume), err
}
