/*
Copyright 2014 Google Inc. All rights reserved.

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

package client

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// FakeLimitRanges implements PodsInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakeLimitRanges struct {
	Fake      *Fake
	Namespace string
}

func (c *FakeLimitRanges) List(selector labels.Selector) (*api.LimitRangeList, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "list-limitRanges"})
	return api.Scheme.CopyOrDie(&c.Fake.LimitRangesList).(*api.LimitRangeList), nil
}

func (c *FakeLimitRanges) Get(name string) (*api.LimitRange, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "get-limitRange", Value: name})
	return &api.LimitRange{NSObjectMeta: api.NSObjectMeta{ObjectMeta: api.ObjectMeta{Name: name}, Namespace: c.Namespace}}, nil
}

func (c *FakeLimitRanges) Delete(name string) error {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "delete-limitRange", Value: name})
	return nil
}

func (c *FakeLimitRanges) Create(limitRange *api.LimitRange) (*api.LimitRange, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "create-limitRange"})
	return &api.LimitRange{}, nil
}

func (c *FakeLimitRanges) Update(limitRange *api.LimitRange) (*api.LimitRange, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "update-limitRange", Value: limitRange.Name})
	return &api.LimitRange{}, nil
}
