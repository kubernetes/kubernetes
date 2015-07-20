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

// FakePods implements PodsInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakePods struct {
	Fake      *Fake
	Namespace string
}

func (c *FakePods) List(label labels.Selector, field fields.Selector) (*api.PodList, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "list-pods"}, &api.PodList{})
	return obj.(*api.PodList), err
}

func (c *FakePods) Get(name string) (*api.Pod, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "get-pod", Value: name}, &api.Pod{})
	return obj.(*api.Pod), err
}

func (c *FakePods) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(FakeAction{Action: "delete-pod", Value: name}, &api.Pod{})
	return err
}

func (c *FakePods) Create(pod *api.Pod) (*api.Pod, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "create-pod"}, &api.Pod{})
	return obj.(*api.Pod), err
}

func (c *FakePods) Update(pod *api.Pod) (*api.Pod, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "update-pod", Value: pod.Name}, &api.Pod{})
	return obj.(*api.Pod), err
}

func (c *FakePods) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Lock.Lock()
	defer c.Fake.Lock.Unlock()

	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "watch-pods", Value: resourceVersion})
	return c.Fake.Watch, c.Fake.Err
}

func (c *FakePods) Bind(bind *api.Binding) error {
	c.Fake.Lock.Lock()
	defer c.Fake.Lock.Unlock()

	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "bind-pod", Value: bind.Name})
	return nil
}

func (c *FakePods) UpdateStatus(pod *api.Pod) (*api.Pod, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "update-status-pod", Value: pod.Name}, &api.Pod{})
	return obj.(*api.Pod), err
}
