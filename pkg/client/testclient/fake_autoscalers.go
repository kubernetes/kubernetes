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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

type FakeAutoScalers struct {
	Fake      *Fake
	Namespace string
}

func (c *FakeAutoScalers) List(label labels.Selector, field fields.Selector) (*api.AutoScalerList, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "list-autoScalers"}, &api.AutoScalerList{})
	return obj.(*api.AutoScalerList), err
}

func (c *FakeAutoScalers) Create(item *api.AutoScaler) (*api.AutoScaler, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "create-autoScaler"}, &api.AutoScaler{})
	return obj.(*api.AutoScaler), err
}

func (c *FakeAutoScalers) Get(name string) (*api.AutoScaler, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "get-autoScaler", Value: name}, &api.AutoScaler{})
	return obj.(*api.AutoScaler), err
}

func (c *FakeAutoScalers) Update(item *api.AutoScaler) (*api.AutoScaler, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "update-autoScaler", Value: item.Name}, &api.AutoScaler{})
	return obj.(*api.AutoScaler), err
}

func (c *FakeAutoScalers) UpdateStatus(item *api.AutoScaler) (*api.AutoScaler, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "update-status-autoScaler", Value: item}, &api.AutoScaler{})
	return obj.(*api.AutoScaler), err
}

func (c *FakeAutoScalers) Delete(name string) error {
	_, err := c.Fake.Invokes(FakeAction{Action: "delete-autoScaler", Value: name}, &api.AutoScaler{})
	return err
}

func (c *FakeAutoScalers) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "watch-autoScaler", Value: resourceVersion})
	return c.Fake.Watch, c.Fake.Err
}
