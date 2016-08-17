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
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	v1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakePodSecurityPolicies implements PodSecurityPolicyInterface
type FakePodSecurityPolicies struct {
	Fake *FakeExtensions
}

var podsecuritypoliciesResource = unversioned.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "podsecuritypolicies"}

func (c *FakePodSecurityPolicies) Create(podSecurityPolicy *v1beta1.PodSecurityPolicy) (result *v1beta1.PodSecurityPolicy, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootCreateAction(podsecuritypoliciesResource, podSecurityPolicy), &v1beta1.PodSecurityPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.PodSecurityPolicy), err
}

func (c *FakePodSecurityPolicies) Update(podSecurityPolicy *v1beta1.PodSecurityPolicy) (result *v1beta1.PodSecurityPolicy, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateAction(podsecuritypoliciesResource, podSecurityPolicy), &v1beta1.PodSecurityPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.PodSecurityPolicy), err
}

func (c *FakePodSecurityPolicies) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewRootDeleteAction(podsecuritypoliciesResource, name), &v1beta1.PodSecurityPolicy{})
	return err
}

func (c *FakePodSecurityPolicies) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewRootDeleteCollectionAction(podsecuritypoliciesResource, listOptions)

	_, err := c.Fake.Invokes(action, &v1beta1.PodSecurityPolicyList{})
	return err
}

func (c *FakePodSecurityPolicies) Get(name string) (result *v1beta1.PodSecurityPolicy, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootGetAction(podsecuritypoliciesResource, name), &v1beta1.PodSecurityPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.PodSecurityPolicy), err
}

func (c *FakePodSecurityPolicies) List(opts api.ListOptions) (result *v1beta1.PodSecurityPolicyList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootListAction(podsecuritypoliciesResource, opts), &v1beta1.PodSecurityPolicyList{})
	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &v1beta1.PodSecurityPolicyList{}
	for _, item := range obj.(*v1beta1.PodSecurityPolicyList).Items {
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
func (c *FakePodSecurityPolicies) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1beta1.PodSecurityPolicy, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootPatchSubresourceAction(podsecuritypoliciesResource, name, data, subresources...), &v1beta1.PodSecurityPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.PodSecurityPolicy), err
}
