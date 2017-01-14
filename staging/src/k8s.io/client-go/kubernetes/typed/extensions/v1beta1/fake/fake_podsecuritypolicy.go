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
	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	watch "k8s.io/apimachinery/pkg/watch"
	api "k8s.io/client-go/pkg/api"
	v1 "k8s.io/client-go/pkg/api/v1"
	v1beta1 "k8s.io/client-go/pkg/apis/extensions/v1beta1"
	testing "k8s.io/client-go/testing"
)

// FakePodSecurityPolicies implements PodSecurityPolicyInterface
type FakePodSecurityPolicies struct {
	Fake *FakeExtensionsV1beta1
}

var podsecuritypoliciesResource = schema.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "podsecuritypolicies"}

func (c *FakePodSecurityPolicies) Create(podSecurityPolicy *v1beta1.PodSecurityPolicy) (result *v1beta1.PodSecurityPolicy, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(podsecuritypoliciesResource, podSecurityPolicy), &v1beta1.PodSecurityPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.PodSecurityPolicy), err
}

func (c *FakePodSecurityPolicies) Update(podSecurityPolicy *v1beta1.PodSecurityPolicy) (result *v1beta1.PodSecurityPolicy, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(podsecuritypoliciesResource, podSecurityPolicy), &v1beta1.PodSecurityPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.PodSecurityPolicy), err
}

func (c *FakePodSecurityPolicies) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(podsecuritypoliciesResource, name), &v1beta1.PodSecurityPolicy{})
	return err
}

func (c *FakePodSecurityPolicies) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(podsecuritypoliciesResource, listOptions)

	_, err := c.Fake.Invokes(action, &v1beta1.PodSecurityPolicyList{})
	return err
}

func (c *FakePodSecurityPolicies) Get(name string, options meta_v1.GetOptions) (result *v1beta1.PodSecurityPolicy, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(podsecuritypoliciesResource, name), &v1beta1.PodSecurityPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.PodSecurityPolicy), err
}

func (c *FakePodSecurityPolicies) List(opts v1.ListOptions) (result *v1beta1.PodSecurityPolicyList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(podsecuritypoliciesResource, opts), &v1beta1.PodSecurityPolicyList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
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
func (c *FakePodSecurityPolicies) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(podsecuritypoliciesResource, opts))
}

// Patch applies the patch and returns the patched podSecurityPolicy.
func (c *FakePodSecurityPolicies) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1beta1.PodSecurityPolicy, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(podsecuritypoliciesResource, name, data, subresources...), &v1beta1.PodSecurityPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.PodSecurityPolicy), err
}
