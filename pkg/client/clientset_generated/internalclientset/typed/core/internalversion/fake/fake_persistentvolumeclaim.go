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
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	schema "k8s.io/kubernetes/pkg/runtime/schema"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakePersistentVolumeClaims implements PersistentVolumeClaimInterface
type FakePersistentVolumeClaims struct {
	Fake *FakeCore
	ns   string
}

var persistentvolumeclaimsResource = schema.GroupVersionResource{Group: "", Version: "", Resource: "persistentvolumeclaims"}

func (c *FakePersistentVolumeClaims) Create(persistentVolumeClaim *api.PersistentVolumeClaim) (result *api.PersistentVolumeClaim, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(persistentvolumeclaimsResource, c.ns, persistentVolumeClaim), &api.PersistentVolumeClaim{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolumeClaim), err
}

func (c *FakePersistentVolumeClaims) Update(persistentVolumeClaim *api.PersistentVolumeClaim) (result *api.PersistentVolumeClaim, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(persistentvolumeclaimsResource, c.ns, persistentVolumeClaim), &api.PersistentVolumeClaim{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolumeClaim), err
}

func (c *FakePersistentVolumeClaims) UpdateStatus(persistentVolumeClaim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateSubresourceAction(persistentvolumeclaimsResource, "status", c.ns, persistentVolumeClaim), &api.PersistentVolumeClaim{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolumeClaim), err
}

func (c *FakePersistentVolumeClaims) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(persistentvolumeclaimsResource, c.ns, name), &api.PersistentVolumeClaim{})

	return err
}

func (c *FakePersistentVolumeClaims) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(persistentvolumeclaimsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &api.PersistentVolumeClaimList{})
	return err
}

func (c *FakePersistentVolumeClaims) Get(name string) (result *api.PersistentVolumeClaim, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(persistentvolumeclaimsResource, c.ns, name), &api.PersistentVolumeClaim{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolumeClaim), err
}

func (c *FakePersistentVolumeClaims) List(opts api.ListOptions) (result *api.PersistentVolumeClaimList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(persistentvolumeclaimsResource, c.ns, opts), &api.PersistentVolumeClaimList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &api.PersistentVolumeClaimList{}
	for _, item := range obj.(*api.PersistentVolumeClaimList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested persistentVolumeClaims.
func (c *FakePersistentVolumeClaims) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(persistentvolumeclaimsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched persistentVolumeClaim.
func (c *FakePersistentVolumeClaims) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *api.PersistentVolumeClaim, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(persistentvolumeclaimsResource, c.ns, name, data, subresources...), &api.PersistentVolumeClaim{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolumeClaim), err
}
