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

// FakeReplicationControllers implements ReplicationControllerInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeReplicationControllers struct {
	Fake      *Fake
	Namespace string
}

const (
	GetControllerAction    = "get-replicationController"
	UpdateControllerAction = "update-replicationController"
	WatchControllerAction  = "watch-replicationController"
	DeleteControllerAction = "delete-replicationController"
	ListControllerAction   = "list-replicationControllers"
	CreateControllerAction = "create-replicationController"
)

func (c *FakeReplicationControllers) List(selector labels.Selector) (*api.ReplicationControllerList, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: ListControllerAction}, &api.ReplicationControllerList{})
	return obj.(*api.ReplicationControllerList), err
}

func (c *FakeReplicationControllers) Get(name string) (*api.ReplicationController, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: GetControllerAction, Value: name}, &api.ReplicationController{})
	return obj.(*api.ReplicationController), err
}

func (c *FakeReplicationControllers) Create(controller *api.ReplicationController) (*api.ReplicationController, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: CreateControllerAction, Value: controller}, &api.ReplicationController{})
	return obj.(*api.ReplicationController), err
}

func (c *FakeReplicationControllers) Update(controller *api.ReplicationController) (*api.ReplicationController, error) {
	obj, err := c.Fake.Invokes(FakeAction{Action: UpdateControllerAction, Value: controller}, &api.ReplicationController{})
	return obj.(*api.ReplicationController), err
}

func (c *FakeReplicationControllers) Delete(name string) error {
	_, err := c.Fake.Invokes(FakeAction{Action: DeleteControllerAction, Value: name}, &api.ReplicationController{})
	return err
}

func (c *FakeReplicationControllers) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Lock.Lock()
	defer c.Fake.Lock.Unlock()

	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: WatchControllerAction, Value: resourceVersion})
	return c.Fake.Watch, nil
}
