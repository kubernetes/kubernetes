/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/apis/extensions"
	kclientlib "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeNetworkPolicies implements NetworkPolicyInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeNetworkPolicies struct {
	Fake      *FakeExperimental
	Namespace string
}

// Ensure statically that FakeNetworkPolicies implements NetworkPolicyInterface.
var _ kclientlib.NetworkPolicyInterface = &FakeNetworkPolicies{}

func (c *FakeNetworkPolicies) Get(name string) (*extensions.NetworkPolicy, error) {
	obj, err := c.Fake.Invokes(NewGetAction("networkpolicies", c.Namespace, name), &extensions.NetworkPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.NetworkPolicy), err
}

func (c *FakeNetworkPolicies) List(opts api.ListOptions) (*extensions.NetworkPolicyList, error) {
	obj, err := c.Fake.Invokes(NewListAction("networkpolicies", c.Namespace, opts), &extensions.NetworkPolicyList{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.NetworkPolicyList), err
}

func (c *FakeNetworkPolicies) Create(np *extensions.NetworkPolicy) (*extensions.NetworkPolicy, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("networkpolicies", c.Namespace, np), &extensions.NetworkPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.NetworkPolicy), err
}

func (c *FakeNetworkPolicies) Update(np *extensions.NetworkPolicy) (*extensions.NetworkPolicy, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("networkpolicies", c.Namespace, np), &extensions.NetworkPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.NetworkPolicy), err
}

func (c *FakeNetworkPolicies) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewDeleteAction("networkpolicies", c.Namespace, name), &extensions.NetworkPolicy{})
	return err
}

func (c *FakeNetworkPolicies) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("networkpolicies", c.Namespace, opts))
}
