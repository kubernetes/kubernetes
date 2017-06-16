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

// FakePersistentVolumeClaims implements PersistentVolumeClaimInterface
type FakePersistentVolumeClaims struct {
	Fake *FakeCore
	ns   string
}

var persistentvolumeclaimsResource = schema.GroupVersionResource{Group: "", Version: "", Resource: "persistentvolumeclaims"}

var persistentvolumeclaimsKind = schema.GroupVersionKind{Group: "", Version: "", Kind: "PersistentVolumeClaim"}

func (c *FakePersistentVolumeClaims) Create(persistentVolumeClaim *api.PersistentVolumeClaim) (result *api.PersistentVolumeClaim, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(persistentvolumeclaimsResource, c.ns, persistentVolumeClaim), &api.PersistentVolumeClaim{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolumeClaim), err
}

func (c *FakePersistentVolumeClaims) Update(persistentVolumeClaim *api.PersistentVolumeClaim) (result *api.PersistentVolumeClaim, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(persistentvolumeclaimsResource, c.ns, persistentVolumeClaim), &api.PersistentVolumeClaim{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolumeClaim), err
}

func (c *FakePersistentVolumeClaims) UpdateStatus(persistentVolumeClaim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceAction(persistentvolumeclaimsResource, "status", c.ns, persistentVolumeClaim), &api.PersistentVolumeClaim{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolumeClaim), err
}

func (c *FakePersistentVolumeClaims) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(persistentvolumeclaimsResource, c.ns, name), &api.PersistentVolumeClaim{})

	return err
}

func (c *FakePersistentVolumeClaims) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(persistentvolumeclaimsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &api.PersistentVolumeClaimList{})
	return err
}

func (c *FakePersistentVolumeClaims) Get(name string, options v1.GetOptions) (result *api.PersistentVolumeClaim, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(persistentvolumeclaimsResource, c.ns, name), &api.PersistentVolumeClaim{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolumeClaim), err
}

func (c *FakePersistentVolumeClaims) List(opts v1.ListOptions) (result *api.PersistentVolumeClaimList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(persistentvolumeclaimsResource, persistentvolumeclaimsKind, c.ns, opts), &api.PersistentVolumeClaimList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
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
func (c *FakePersistentVolumeClaims) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(persistentvolumeclaimsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched persistentVolumeClaim.
func (c *FakePersistentVolumeClaims) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *api.PersistentVolumeClaim, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(persistentvolumeclaimsResource, c.ns, name, data, subresources...), &api.PersistentVolumeClaim{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.PersistentVolumeClaim), err
}
