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

// FakePods implements PodsInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakePods struct {
	Fake      *Fake
	Namespace string
}

func (c *FakePods) List(selector labels.Selector) (*api.PodList, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "list-pods"})
	return api.Scheme.CopyOrDie(&c.Fake.PodsList).(*api.PodList), nil
}

func (c *FakePods) Get(name string) (*api.Pod, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "get-pod", Value: name})
	return &api.Pod{ObjectMeta: api.ObjectMeta{Name: name, Namespace: c.Namespace}}, nil
}

func (c *FakePods) Delete(name string) error {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "delete-pod", Value: name})
	return nil
}

func (c *FakePods) Create(pod *api.Pod) (*api.Pod, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "create-pod"})
	return &api.Pod{}, nil
}

func (c *FakePods) Update(pod *api.Pod) (*api.Pod, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "update-pod", Value: pod.Name})
	return &api.Pod{}, nil
}
