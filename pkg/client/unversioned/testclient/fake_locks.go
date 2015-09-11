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
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeLocks implements LockInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakeLocks struct {
	Fake      *FakeExperimental
	Namespace string
}

func (c *FakeLocks) Create(lock *expapi.Lock) (*expapi.Lock, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("locks", c.Namespace, lock), lock)
	if obj == nil {
		return nil, err
	}
	return obj.(*expapi.Lock), err
}

func (c *FakeLocks) List(label labels.Selector) (*expapi.LockList, error) {
	obj, err := c.Fake.Invokes(NewListAction("locks", c.Namespace, label, nil), &expapi.LockList{})
	if obj == nil {
		return nil, err
	}
	list := &expapi.LockList{}
	for _, lock := range obj.(*expapi.LockList).Items {
		if label.Matches(labels.Set(lock.Labels)) {
			list.Items = append(list.Items, lock)
		}
	}

	return list, err
}

func (c *FakeLocks) Get(name string) (*expapi.Lock, error) {
	obj, err := c.Fake.Invokes(NewGetAction("locks", c.Namespace, name), &expapi.Lock{})
	if obj == nil {
		return nil, err
	}
	return obj.(*expapi.Lock), err
}

func (c *FakeLocks) Update(lock *expapi.Lock) (*expapi.Lock, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("locks", c.Namespace, lock), lock)
	if obj == nil {
		return nil, err
	}
	return obj.(*expapi.Lock), err
}

func (c *FakeLocks) Delete(name string) error {
	_, err := c.Fake.Invokes(NewDeleteAction("locks", c.Namespace, name), &expapi.Lock{})
	return err
}

func (c *FakeLocks) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("locks", c.Namespace, label, field, resourceVersion))
}
