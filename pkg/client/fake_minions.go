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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
)

// FakeNodes implements MinionInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeNodes struct {
	Fake *Fake
}

func (c *FakeNodes) Get(name string) (*api.Node, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "get-minion", Value: name})
	for i := range c.Fake.MinionsList.Items {
		if c.Fake.MinionsList.Items[i].Name == name {
			return &c.Fake.MinionsList.Items[i], nil
		}
	}
	return nil, errors.NewNotFound("Minions", name)
}

func (c *FakeNodes) List() (*api.NodeList, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "list-minions", Value: nil})
	return &c.Fake.MinionsList, nil
}

func (c *FakeNodes) Create(minion *api.Node) (*api.Node, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "create-minion", Value: minion})
	return &api.Node{}, nil
}

func (c *FakeNodes) Delete(id string) error {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "delete-minion", Value: id})
	return nil
}

func (c *FakeNodes) Update(minion *api.Node) (*api.Node, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "update-minion", Value: minion})
	return &api.Node{}, nil
}
