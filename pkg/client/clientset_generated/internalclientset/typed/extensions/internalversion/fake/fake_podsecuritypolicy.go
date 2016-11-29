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
	extensions "k8s.io/kubernetes/pkg/apis/extensions"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	schema "k8s.io/kubernetes/pkg/runtime/schema"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakePodSecurityPolicies implements PodSecurityPolicyInterface
type FakePodSecurityPolicies struct {
	Fake *FakeExtensions
}

var podsecuritypoliciesResource = schema.GroupVersionResource{Group: "extensions", Version: "", Resource: "podsecuritypolicies"}

func (c *FakePodSecurityPolicies) Create(podSecurityPolicy *extensions.PodSecurityPolicy) (result *extensions.PodSecurityPolicy, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootCreateAction(podsecuritypoliciesResource, podSecurityPolicy), &extensions.PodSecurityPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.PodSecurityPolicy), err
}

func (c *FakePodSecurityPolicies) Update(podSecurityPolicy *extensions.PodSecurityPolicy) (result *extensions.PodSecurityPolicy, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateAction(podsecuritypoliciesResource, podSecurityPolicy), &extensions.PodSecurityPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.PodSecurityPolicy), err
}

func (c *FakePodSecurityPolicies) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewRootDeleteAction(podsecuritypoliciesResource, name), &extensions.PodSecurityPolicy{})
	return err
}

func (c *FakePodSecurityPolicies) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewRootDeleteCollectionAction(podsecuritypoliciesResource, listOptions)

	_, err := c.Fake.Invokes(action, &extensions.PodSecurityPolicyList{})
	return err
}

func (c *FakePodSecurityPolicies) Get(name string) (result *extensions.PodSecurityPolicy, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootGetAction(podsecuritypoliciesResource, name), &extensions.PodSecurityPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.PodSecurityPolicy), err
}

func (c *FakePodSecurityPolicies) List(opts api.ListOptions) (result *extensions.PodSecurityPolicyList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootListAction(podsecuritypoliciesResource, opts), &extensions.PodSecurityPolicyList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &extensions.PodSecurityPolicyList{}
	for _, item := range obj.(*extensions.PodSecurityPolicyList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested podSecurityPolicies.
func (c *FakePodSecurityPolicies) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewRootWatchAction(podsecuritypoliciesResource, opts))
}

// Patch applies the patch and returns the patched podSecurityPolicy.
func (c *FakePodSecurityPolicies) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *extensions.PodSecurityPolicy, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootPatchSubresourceAction(podsecuritypoliciesResource, name, data, subresources...), &extensions.PodSecurityPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.PodSecurityPolicy), err
}
