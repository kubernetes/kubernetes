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
	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	v1 "k8s.io/client-go/pkg/api/v1"
	testing "k8s.io/client-go/testing"
)

// FakePersistentVolumes implements PersistentVolumeInterface
type FakePersistentVolumes struct {
	Fake *FakeCoreV1
}

var persistentvolumesResource = schema.GroupVersionResource{Group: "", Version: "v1", Resource: "persistentvolumes"}

func (c *FakePersistentVolumes) Create(persistentVolume *v1.PersistentVolume) (result *v1.PersistentVolume, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(persistentvolumesResource, persistentVolume), &v1.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.PersistentVolume), err
}

func (c *FakePersistentVolumes) Update(persistentVolume *v1.PersistentVolume) (result *v1.PersistentVolume, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(persistentvolumesResource, persistentVolume), &v1.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.PersistentVolume), err
}

func (c *FakePersistentVolumes) UpdateStatus(persistentVolume *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateSubresourceAction(persistentvolumesResource, "status", persistentVolume), &v1.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.PersistentVolume), err
}

func (c *FakePersistentVolumes) Delete(name string, options *meta_v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(persistentvolumesResource, name), &v1.PersistentVolume{})
	return err
}

func (c *FakePersistentVolumes) DeleteCollection(options *meta_v1.DeleteOptions, listOptions meta_v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(persistentvolumesResource, listOptions)

	_, err := c.Fake.Invokes(action, &v1.PersistentVolumeList{})
	return err
}

func (c *FakePersistentVolumes) Get(name string, options meta_v1.GetOptions) (result *v1.PersistentVolume, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(persistentvolumesResource, name), &v1.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.PersistentVolume), err
}

func (c *FakePersistentVolumes) List(opts meta_v1.ListOptions) (result *v1.PersistentVolumeList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(persistentvolumesResource, opts), &v1.PersistentVolumeList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1.PersistentVolumeList{}
	for _, item := range obj.(*v1.PersistentVolumeList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested persistentVolumes.
func (c *FakePersistentVolumes) Watch(opts meta_v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(persistentvolumesResource, opts))
}

// Patch applies the patch and returns the patched persistentVolume.
func (c *FakePersistentVolumes) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1.PersistentVolume, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(persistentvolumesResource, name, data, subresources...), &v1.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1.PersistentVolume), err
}
