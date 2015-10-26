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

// FakeReplicationControllers implements ReplicationControllerInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeReplicationControllers struct {
	Fake      *Fake
	Namespace string
}

func (c *FakeReplicationControllers) Get(name string) (*api.ReplicationController, error) {
	obj, err := c.Fake.Invokes(NewGetAction("replicationcontrollers", c.Namespace, name), &api.ReplicationController{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.ReplicationController), err
}

func (c *FakeReplicationControllers) List(label labels.Selector, field fields.Selector) (*api.ReplicationControllerList, error) {
	obj, err := c.Fake.Invokes(NewListAction("replicationcontrollers", c.Namespace, label, field), &api.ReplicationControllerList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.ReplicationControllerList), err
}

func (c *FakeReplicationControllers) Create(controller *api.ReplicationController) (*api.ReplicationController, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("replicationcontrollers", c.Namespace, controller), controller)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.ReplicationController), err
}

func (c *FakeReplicationControllers) Update(controller *api.ReplicationController) (*api.ReplicationController, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("replicationcontrollers", c.Namespace, controller), controller)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.ReplicationController), err
}

func (c *FakeReplicationControllers) UpdateStatus(controller *api.ReplicationController) (*api.ReplicationController, error) {
	obj, err := c.Fake.Invokes(NewUpdateSubresourceAction("replicationcontrollers", "status", c.Namespace, controller), &api.ReplicationController{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.ReplicationController), err
}

func (c *FakeReplicationControllers) Delete(name string) error {
	_, err := c.Fake.Invokes(NewDeleteAction("replicationcontrollers", c.Namespace, name), &api.ReplicationController{})
	return err
}

func (c *FakeReplicationControllers) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("replicationcontrollers", c.Namespace, label, field, resourceVersion))
}
