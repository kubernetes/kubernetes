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

package testclient

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/watch"
)

// FakePetSets implements PetSetsInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakePetSets struct {
	Fake      *FakeApps
	Namespace string
}

func (c *FakePetSets) Get(name string) (*apps.PetSet, error) {
	obj, err := c.Fake.Invokes(NewGetAction("petsets", c.Namespace, name), &apps.PetSet{})
	if obj == nil {
		return nil, err
	}

	return obj.(*apps.PetSet), err
}

func (c *FakePetSets) List(opts api.ListOptions) (*apps.PetSetList, error) {
	obj, err := c.Fake.Invokes(NewListAction("petsets", c.Namespace, opts), &apps.PetSetList{})
	if obj == nil {
		return nil, err
	}
	return obj.(*apps.PetSetList), err
}

func (c *FakePetSets) Create(rs *apps.PetSet) (*apps.PetSet, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("petsets", c.Namespace, rs), rs)
	if obj == nil {
		return nil, err
	}

	return obj.(*apps.PetSet), err
}

func (c *FakePetSets) Update(rs *apps.PetSet) (*apps.PetSet, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("petsets", c.Namespace, rs), rs)
	if obj == nil {
		return nil, err
	}

	return obj.(*apps.PetSet), err
}

func (c *FakePetSets) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewDeleteAction("petsets", c.Namespace, name), &apps.PetSet{})
	return err
}

func (c *FakePetSets) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("petsets", c.Namespace, opts))
}

func (c *FakePetSets) UpdateStatus(rs *apps.PetSet) (result *apps.PetSet, err error) {
	obj, err := c.Fake.Invokes(NewUpdateSubresourceAction("petsets", "status", c.Namespace, rs), rs)
	if obj == nil {
		return nil, err
	}

	return obj.(*apps.PetSet), err
}
