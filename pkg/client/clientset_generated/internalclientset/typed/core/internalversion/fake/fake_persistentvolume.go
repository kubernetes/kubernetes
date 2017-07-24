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
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	testing "k8s.io/client-go/testing"
	api "k8s.io/kubernetes/pkg/api"
)

// FakePersistentVolumes implements PersistentVolumeInterface
type FakePersistentVolumes struct {
	Fake *FakeCore
}

var persistentvolumesResource = schema.GroupVersionResource{Group: "", Version: "", Resource: "persistentvolumes"}

var persistentvolumesKind = schema.GroupVersionKind{Group: "", Version: "", Kind: "PersistentVolume"}

// Get takes name of the persistentVolume, and returns the corresponding persistentVolume object, and an error if there is any.
func (c *FakePersistentVolumes) Get(name string, options v1.GetOptions) (result *api.PersistentVolume, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(persistentvolumesResource, name), &api.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolume), err
}

// List takes label and field selectors, and returns the list of PersistentVolumes that match those selectors.
func (c *FakePersistentVolumes) List(opts v1.ListOptions) (result *api.PersistentVolumeList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(persistentvolumesResource, persistentvolumesKind, opts), &api.PersistentVolumeList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
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
func (c *FakePersistentVolumes) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(persistentvolumesResource, opts))
}

// Create takes the representation of a persistentVolume and creates it.  Returns the server's representation of the persistentVolume, and an error, if there is any.
func (c *FakePersistentVolumes) Create(persistentVolume *api.PersistentVolume) (result *api.PersistentVolume, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(persistentvolumesResource, persistentVolume), &api.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolume), err
}

// Update takes the representation of a persistentVolume and updates it. Returns the server's representation of the persistentVolume, and an error, if there is any.
func (c *FakePersistentVolumes) Update(persistentVolume *api.PersistentVolume) (result *api.PersistentVolume, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(persistentvolumesResource, persistentVolume), &api.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolume), err
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().
func (c *FakePersistentVolumes) UpdateStatus(persistentVolume *api.PersistentVolume) (*api.PersistentVolume, error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateSubresourceAction(persistentvolumesResource, "status", persistentVolume), &api.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolume), err
}

// Delete takes name of the persistentVolume and deletes it. Returns an error if one occurs.
func (c *FakePersistentVolumes) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(persistentvolumesResource, name), &api.PersistentVolume{})
	return err
}

// DeleteCollection deletes a collection of objects.
func (c *FakePersistentVolumes) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(persistentvolumesResource, listOptions)

	_, err := c.Fake.Invokes(action, &api.PersistentVolumeList{})
	return err
}

// Patch applies the patch and returns the patched persistentVolume.
func (c *FakePersistentVolumes) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *api.PersistentVolume, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(persistentvolumesResource, name, data, subresources...), &api.PersistentVolume{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolume), err
}
