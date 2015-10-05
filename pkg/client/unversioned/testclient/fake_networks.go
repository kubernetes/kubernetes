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

// FakeNetworks implements NetworksInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakeNetworks struct {
	Fake *Fake
}

func (c *FakeNetworks) Get(name string) (*api.Network, error) {
	obj, err := c.Fake.Invokes(NewRootGetAction("networks", name), &api.Network{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Network), err
}

func (c *FakeNetworks) List(label labels.Selector, field fields.Selector) (*api.NetworkList, error) {
	obj, err := c.Fake.Invokes(NewRootListAction("networks", label, field), &api.NetworkList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.NetworkList), err
}

func (c *FakeNetworks) Create(network *api.Network) (*api.Network, error) {
	obj, err := c.Fake.Invokes(NewRootCreateAction("networks", network), network)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Network), err
}

func (c *FakeNetworks) Update(network *api.Network) (*api.Network, error) {
	obj, err := c.Fake.Invokes(NewRootUpdateAction("networks", network), network)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Network), err
}

func (c *FakeNetworks) Delete(name string) error {
	_, err := c.Fake.Invokes(NewRootDeleteAction("networks", name), &api.Network{})
	return err
}

func (c *FakeNetworks) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewRootWatchAction("networks", label, field, resourceVersion))
}

func (c *FakeNetworks) Finalize(network *api.Network) (*api.Network, error) {
	action := CreateActionImpl{}
	action.Verb = "create"
	action.Resource = "networks"
	action.Subresource = "finalize"
	action.Object = network

	obj, err := c.Fake.Invokes(action, network)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Network), err
}

func (c *FakeNetworks) Status(network *api.Network) (*api.Network, error) {
	action := CreateActionImpl{}
	action.Verb = "create"
	action.Resource = "networks"
	action.Subresource = "status"
	action.Object = network

	obj, err := c.Fake.Invokes(action, network)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Network), err
}
