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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// FakeReplicationControllers implements ReplicationControllerInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeReplicationControllers struct {
	Fake      *Fake
	Namespace string
}

func (c *FakeReplicationControllers) List(selector labels.Selector) (*api.ReplicationControllerList, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "list-controllers"})
	return &api.ReplicationControllerList{}, nil
}

func (c *FakeReplicationControllers) Get(name string) (*api.ReplicationController, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "get-controller", Value: name})
	return api.Scheme.CopyOrDie(&c.Fake.Ctrl).(*api.ReplicationController), nil
}

func (c *FakeReplicationControllers) Create(controller *api.ReplicationController) (*api.ReplicationController, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "create-controller", Value: controller})
	return &api.ReplicationController{}, nil
}

func (c *FakeReplicationControllers) Update(controller *api.ReplicationController) (*api.ReplicationController, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "update-controller", Value: controller})
	return &api.ReplicationController{}, nil
}

func (c *FakeReplicationControllers) Delete(controller string) error {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "delete-controller", Value: controller})
	return nil
}

func (c *FakeReplicationControllers) Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "watch-controllers", Value: resourceVersion})
	return c.Fake.Watch, nil
}
