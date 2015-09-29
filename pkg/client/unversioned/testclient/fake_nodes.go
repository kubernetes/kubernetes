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

// FakeNodes implements NodeInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeNodes struct {
	Fake *Fake
}

func (c *FakeNodes) Get(name string) (*api.Node, error) {
	obj, err := c.Fake.Invokes(NewRootGetAction("nodes", name), &api.Node{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Node), err
}

func (c *FakeNodes) List(label labels.Selector, field fields.Selector) (*api.NodeList, error) {
	obj, err := c.Fake.Invokes(NewRootListAction("nodes", label, field), &api.NodeList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.NodeList), err
}

func (c *FakeNodes) Create(node *api.Node) (*api.Node, error) {
	obj, err := c.Fake.Invokes(NewRootCreateAction("nodes", node), node)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Node), err
}

func (c *FakeNodes) Update(node *api.Node) (*api.Node, error) {
	obj, err := c.Fake.Invokes(NewRootUpdateAction("nodes", node), node)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Node), err
}

func (c *FakeNodes) Delete(name string) error {
	_, err := c.Fake.Invokes(NewRootDeleteAction("nodes", name), &api.Node{})
	return err
}

func (c *FakeNodes) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewRootWatchAction("nodes", label, field, resourceVersion))
}

func (c *FakeNodes) UpdateStatus(node *api.Node) (*api.Node, error) {
	action := CreateActionImpl{}
	action.Verb = "update"
	action.Resource = "nodes"
	action.Subresource = "status"
	action.Object = node

	obj, err := c.Fake.Invokes(action, node)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Node), err
}
