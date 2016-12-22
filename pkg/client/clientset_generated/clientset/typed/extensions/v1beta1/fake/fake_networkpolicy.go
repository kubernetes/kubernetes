/*
Copyright 2016 The Kubernetes Authors.

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
	api "k8s.io/kubernetes/pkg/api"
	v1 "k8s.io/kubernetes/pkg/api/v1"
	v1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	meta_v1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	schema "k8s.io/kubernetes/pkg/runtime/schema"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeNetworkPolicies implements NetworkPolicyInterface
type FakeNetworkPolicies struct {
	Fake *FakeExtensionsV1beta1
	ns   string
}

var networkpoliciesResource = schema.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "networkpolicies"}

func (c *FakeNetworkPolicies) Create(networkPolicy *v1beta1.NetworkPolicy) (result *v1beta1.NetworkPolicy, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(networkpoliciesResource, c.ns, networkPolicy), &v1beta1.NetworkPolicy{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.NetworkPolicy), err
}

func (c *FakeNetworkPolicies) Update(networkPolicy *v1beta1.NetworkPolicy) (result *v1beta1.NetworkPolicy, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(networkpoliciesResource, c.ns, networkPolicy), &v1beta1.NetworkPolicy{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.NetworkPolicy), err
}

func (c *FakeNetworkPolicies) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(networkpoliciesResource, c.ns, name), &v1beta1.NetworkPolicy{})

	return err
}

func (c *FakeNetworkPolicies) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := core.NewDeleteCollectionAction(networkpoliciesResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1beta1.NetworkPolicyList{})
	return err
}

func (c *FakeNetworkPolicies) Get(name string, options meta_v1.GetOptions) (result *v1beta1.NetworkPolicy, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(networkpoliciesResource, c.ns, name), &v1beta1.NetworkPolicy{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.NetworkPolicy), err
}

func (c *FakeNetworkPolicies) List(opts v1.ListOptions) (result *v1beta1.NetworkPolicyList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(networkpoliciesResource, c.ns, opts), &v1beta1.NetworkPolicyList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1beta1.NetworkPolicyList{}
	for _, item := range obj.(*v1beta1.NetworkPolicyList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested networkPolicies.
func (c *FakeNetworkPolicies) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(networkpoliciesResource, c.ns, opts))

}

// Patch applies the patch and returns the patched networkPolicy.
func (c *FakeNetworkPolicies) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1beta1.NetworkPolicy, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(networkpoliciesResource, c.ns, name, data, subresources...), &v1beta1.NetworkPolicy{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.NetworkPolicy), err
}
