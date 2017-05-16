/*
Copyright 2017 The Kubernetes Authors.

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

package fake

import (
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	audit "k8s.io/apiserver/pkg/apis/audit"
	testing "k8s.io/client-go/testing"
)

// FakePolicies implements PolicyInterface
type FakePolicies struct {
	Fake *FakeAudit
}

var policiesResource = schema.GroupVersionResource{Group: "audit.k8s.io", Version: "", Resource: "policies"}

var policiesKind = schema.GroupVersionKind{Group: "audit.k8s.io", Version: "", Kind: "Policy"}

func (c *FakePolicies) Create(policy *audit.Policy) (result *audit.Policy, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(policiesResource, policy), &audit.Policy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*audit.Policy), err
}

func (c *FakePolicies) Update(policy *audit.Policy) (result *audit.Policy, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(policiesResource, policy), &audit.Policy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*audit.Policy), err
}

func (c *FakePolicies) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(policiesResource, name), &audit.Policy{})
	return err
}

func (c *FakePolicies) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(policiesResource, listOptions)

	_, err := c.Fake.Invokes(action, &audit.PolicyList{})
	return err
}

func (c *FakePolicies) Get(name string, options v1.GetOptions) (result *audit.Policy, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(policiesResource, name), &audit.Policy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*audit.Policy), err
}

func (c *FakePolicies) List(opts v1.ListOptions) (result *audit.PolicyList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(policiesResource, policiesKind, opts), &audit.PolicyList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &audit.PolicyList{}
	for _, item := range obj.(*audit.PolicyList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested policies.
func (c *FakePolicies) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(policiesResource, opts))
}

// Patch applies the patch and returns the patched policy.
func (c *FakePolicies) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *audit.Policy, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(policiesResource, name, data, subresources...), &audit.Policy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*audit.Policy), err
}
