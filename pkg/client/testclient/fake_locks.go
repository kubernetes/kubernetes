/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeLocks implements LockInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakeLocks struct {
	Fake      *Fake
	Namespace string
}

func (c *FakeLocks) Create(lock *api.Lock) (*api.Lock, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "create-lock"}, &api.Lock{})
	return obj.(*api.Lock), err
}

func (c *FakeLocks) List(selector labels.Selector) (*api.LockList, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "list-locks"}, &api.LockList{})
	return obj.(*api.LockList), err
}

func (c *FakeLocks) Get(name string) (*api.Lock, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "get-lock", Value: name}, &api.Lock{})
	return obj.(*api.Lock), err
}

func (c *FakeLocks) Update(lock *api.Lock) (*api.Lock, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "update-lock", Value: lock.Name}, &api.Lock{})
	return obj.(*api.Lock), err
}

func (c *FakeLocks) Delete(name string) error {
	_, err := c.Fake.Invokes(FakeAction{Action: "delete-lock", Value: name}, &api.Lock{})
	return err
}

func (c *FakeLocks) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "watch-lock", Value: resourceVersion})
	return c.Fake.Watch, nil
}
