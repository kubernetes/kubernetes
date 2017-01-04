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
	api "k8s.io/kubernetes/pkg/api"
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	storage "k8s.io/kubernetes/pkg/apis/storage"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeStorageClasses implements StorageClassInterface
type FakeStorageClasses struct {
	Fake *FakeStorage
}

var storageclassesResource = unversioned.GroupVersionResource{Group: "storage.k8s.io", Version: "", Resource: "storageclasses"}

func (c *FakeStorageClasses) Create(storageClass *storage.StorageClass) (result *storage.StorageClass, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootCreateAction(storageclassesResource, storageClass), &storage.StorageClass{})
	if obj == nil {
		return nil, err
	}
	return obj.(*storage.StorageClass), err
}

func (c *FakeStorageClasses) Update(storageClass *storage.StorageClass) (result *storage.StorageClass, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateAction(storageclassesResource, storageClass), &storage.StorageClass{})
	if obj == nil {
		return nil, err
	}
	return obj.(*storage.StorageClass), err
}

func (c *FakeStorageClasses) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewRootDeleteAction(storageclassesResource, name), &storage.StorageClass{})
	return err
}

func (c *FakeStorageClasses) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewRootDeleteCollectionAction(storageclassesResource, listOptions)

	_, err := c.Fake.Invokes(action, &storage.StorageClassList{})
	return err
}

func (c *FakeStorageClasses) Get(name string) (result *storage.StorageClass, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootGetAction(storageclassesResource, name), &storage.StorageClass{})
	if obj == nil {
		return nil, err
	}
	return obj.(*storage.StorageClass), err
}

func (c *FakeStorageClasses) List(opts api.ListOptions) (result *storage.StorageClassList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootListAction(storageclassesResource, opts), &storage.StorageClassList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &storage.StorageClassList{}
	for _, item := range obj.(*storage.StorageClassList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested storageClasses.
func (c *FakeStorageClasses) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewRootWatchAction(storageclassesResource, opts))
}

// Patch applies the patch and returns the patched storageClass.
func (c *FakeStorageClasses) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *storage.StorageClass, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootPatchSubresourceAction(storageclassesResource, name, data, subresources...), &storage.StorageClass{})
	if obj == nil {
		return nil, err
	}
	return obj.(*storage.StorageClass), err
}
