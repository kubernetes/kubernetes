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
	v1 "k8s.io/kubernetes/pkg/api/v1"
	v1alpha1 "k8s.io/kubernetes/pkg/apis/apps/v1alpha1"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakePetSets implements PetSetInterface
type FakePetSets struct {
	Fake *FakeApps
	ns   string
}

var petsetsResource = unversioned.GroupVersionResource{Group: "apps", Version: "v1alpha1", Resource: "petsets"}

func (c *FakePetSets) Create(petSet *v1alpha1.PetSet) (result *v1alpha1.PetSet, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(petsetsResource, c.ns, petSet), &v1alpha1.PetSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.PetSet), err
}

func (c *FakePetSets) Update(petSet *v1alpha1.PetSet) (result *v1alpha1.PetSet, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(petsetsResource, c.ns, petSet), &v1alpha1.PetSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.PetSet), err
}

func (c *FakePetSets) UpdateStatus(petSet *v1alpha1.PetSet) (*v1alpha1.PetSet, error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateSubresourceAction(petsetsResource, "status", c.ns, petSet), &v1alpha1.PetSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.PetSet), err
}

func (c *FakePetSets) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(petsetsResource, c.ns, name), &v1alpha1.PetSet{})

	return err
}

func (c *FakePetSets) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := core.NewDeleteCollectionAction(petsetsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1alpha1.PetSetList{})
	return err
}

func (c *FakePetSets) Get(name string) (result *v1alpha1.PetSet, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(petsetsResource, c.ns, name), &v1alpha1.PetSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.PetSet), err
}

func (c *FakePetSets) List(opts v1.ListOptions) (result *v1alpha1.PetSetList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(petsetsResource, c.ns, opts), &v1alpha1.PetSetList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1alpha1.PetSetList{}
	for _, item := range obj.(*v1alpha1.PetSetList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested petSets.
func (c *FakePetSets) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(petsetsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched petSet.
func (c *FakePetSets) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1alpha1.PetSet, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(petsetsResource, c.ns, name, data, subresources...), &v1alpha1.PetSet{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.PetSet), err
}
