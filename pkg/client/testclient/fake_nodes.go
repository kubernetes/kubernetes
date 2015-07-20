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

// FakeNodes implements MinionInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeNodes struct {
	Fake *Fake
}

func (c *FakeNodes) Get(name string) (*api.Node, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "get-node", Value: name}, &api.Node{})
	return obj.(*api.Node), err
}

func (c *FakeNodes) List(label labels.Selector, field fields.Selector) (*api.NodeList, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "list-nodes"}, &api.NodeList{})
	return obj.(*api.NodeList), err
}

func (c *FakeNodes) Create(minion *api.Node) (*api.Node, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "create-node", Value: minion}, &api.Node{})
	return obj.(*api.Node), err
}

func (c *FakeNodes) Delete(name string) error {
	_, err := c.Fake.Invokes(FakeAction{Action: "delete-node", Value: name}, &api.Node{})
	return err
}

func (c *FakeNodes) Update(minion *api.Node) (*api.Node, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "update-node", Value: minion}, &api.Node{})
	return obj.(*api.Node), err
}

func (c *FakeNodes) UpdateStatus(minion *api.Node) (*api.Node, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: "update-status-node", Value: minion}, &api.Node{})
	return obj.(*api.Node), err
}

func (c *FakeNodes) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Lock.Lock()
	defer c.Fake.Lock.Unlock()

	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "watch-nodes", Value: resourceVersion})
	return c.Fake.Watch, c.Fake.Err
}
