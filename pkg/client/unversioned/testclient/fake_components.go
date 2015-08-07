/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

// FakeComponentsClient implements ComponentsClient.
type FakeComponentsClient struct {
	Fake *Fake
}

func (c *FakeComponentsClient) Get(name string) (*api.Component, error) {
	obj, err := c.Fake.Invokes(NewRootGetAction("components", name), &api.Component{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Component), err
}

func (c *FakeComponentsClient) List(label labels.Selector, field fields.Selector) (*api.ComponentList, error) {
	obj, err := c.Fake.Invokes(NewRootListAction("components", label, field), &api.ComponentList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.ComponentList), err
}

func (c *FakeComponentsClient) Create(minion *api.Component) (*api.Component, error) {
	obj, err := c.Fake.Invokes(NewRootCreateAction("components", minion), minion)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Component), err
}

func (c *FakeComponentsClient) Update(minion *api.Component) (*api.Component, error) {
	obj, err := c.Fake.Invokes(NewRootUpdateAction("components", minion), minion)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Component), err
}

func (c *FakeComponentsClient) Delete(name string) error {
	_, err := c.Fake.Invokes(NewRootDeleteAction("components", name), &api.Component{})
	return err
}

func (c *FakeComponentsClient) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Invokes(NewRootWatchAction("components", label, field, resourceVersion), nil)
	return c.Fake.Watch, c.Fake.Err()
}

func (c *FakeComponentsClient) UpdateStatus(minion *api.Component) (*api.Component, error) {
	action := CreateActionImpl{}
	action.Verb = "update"
	action.Resource = "components"
	action.Subresource = "status"
	action.Object = minion

	obj, err := c.Fake.Invokes(action, minion)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Component), err
}
