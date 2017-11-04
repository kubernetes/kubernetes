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
	testing "k8s.io/client-go/testing"
	extensions "k8s.io/kubernetes/pkg/apis/extensions"
)

// FakePodSecurityPolicies implements PodSecurityPolicyInterface
type FakePodSecurityPolicies struct {
	Fake *FakeExtensions
}

var podsecuritypoliciesResource = schema.GroupVersionResource{Group: "extensions", Version: "", Resource: "podsecuritypolicies"}

var podsecuritypoliciesKind = schema.GroupVersionKind{Group: "extensions", Version: "", Kind: "PodSecurityPolicy"}

// Get takes name of the podSecurityPolicy, and returns the corresponding podSecurityPolicy object, and an error if there is any.
func (c *FakePodSecurityPolicies) Get(name string, options v1.GetOptions) (result *extensions.PodSecurityPolicy, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(podsecuritypoliciesResource, name), &extensions.PodSecurityPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.PodSecurityPolicy), err
}

// List takes label and field selectors, and returns the list of PodSecurityPolicies that match those selectors.
func (c *FakePodSecurityPolicies) List(opts v1.ListOptions) (result *extensions.PodSecurityPolicyList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(podsecuritypoliciesResource, podsecuritypoliciesKind, opts), &extensions.PodSecurityPolicyList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
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
func (c *FakePodSecurityPolicies) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(podsecuritypoliciesResource, opts))
}

// Create takes the representation of a podSecurityPolicy and creates it.  Returns the server's representation of the podSecurityPolicy, and an error, if there is any.
func (c *FakePodSecurityPolicies) Create(podSecurityPolicy *extensions.PodSecurityPolicy) (result *extensions.PodSecurityPolicy, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(podsecuritypoliciesResource, podSecurityPolicy), &extensions.PodSecurityPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.PodSecurityPolicy), err
}

// Update takes the representation of a podSecurityPolicy and updates it. Returns the server's representation of the podSecurityPolicy, and an error, if there is any.
func (c *FakePodSecurityPolicies) Update(podSecurityPolicy *extensions.PodSecurityPolicy) (result *extensions.PodSecurityPolicy, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(podsecuritypoliciesResource, podSecurityPolicy), &extensions.PodSecurityPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.PodSecurityPolicy), err
}

// Delete takes name of the podSecurityPolicy and deletes it. Returns an error if one occurs.
func (c *FakePodSecurityPolicies) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(podsecuritypoliciesResource, name), &extensions.PodSecurityPolicy{})
	return err
}

// DeleteCollection deletes a collection of objects.
func (c *FakePodSecurityPolicies) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(podsecuritypoliciesResource, listOptions)

	_, err := c.Fake.Invokes(action, &extensions.PodSecurityPolicyList{})
	return err
}

// Patch applies the patch and returns the patched podSecurityPolicy.
func (c *FakePodSecurityPolicies) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *extensions.PodSecurityPolicy, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(podsecuritypoliciesResource, name, data, subresources...), &extensions.PodSecurityPolicy{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.PodSecurityPolicy), err
}
