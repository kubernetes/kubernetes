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
	v1alpha1 "k8s.io/kubernetes/pkg/apis/apps/v1alpha1"
)

// FakePodInjectionPolicies implements PodInjectionPolicyInterface
type FakePodInjectionPolicies struct {
	Fake *FakeAppsV1alpha1
	ns   string
}

var podinjectionpoliciesResource = schema.GroupVersionResource{Group: "apps", Version: "v1alpha1", Resource: "podinjectionpolicies"}

func (c *FakePodInjectionPolicies) Create(podInjectionPolicy *v1alpha1.PodInjectionPolicy) (result *v1alpha1.PodInjectionPolicy, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(podinjectionpoliciesResource, c.ns, podInjectionPolicy), &v1alpha1.PodInjectionPolicy{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.PodInjectionPolicy), err
}

func (c *FakePodInjectionPolicies) Update(podInjectionPolicy *v1alpha1.PodInjectionPolicy) (result *v1alpha1.PodInjectionPolicy, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(podinjectionpoliciesResource, c.ns, podInjectionPolicy), &v1alpha1.PodInjectionPolicy{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.PodInjectionPolicy), err
}

func (c *FakePodInjectionPolicies) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(podinjectionpoliciesResource, c.ns, name), &v1alpha1.PodInjectionPolicy{})

	return err
}

func (c *FakePodInjectionPolicies) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(podinjectionpoliciesResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1alpha1.PodInjectionPolicyList{})
	return err
}

func (c *FakePodInjectionPolicies) Get(name string, options v1.GetOptions) (result *v1alpha1.PodInjectionPolicy, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(podinjectionpoliciesResource, c.ns, name), &v1alpha1.PodInjectionPolicy{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.PodInjectionPolicy), err
}

func (c *FakePodInjectionPolicies) List(opts v1.ListOptions) (result *v1alpha1.PodInjectionPolicyList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(podinjectionpoliciesResource, c.ns, opts), &v1alpha1.PodInjectionPolicyList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1alpha1.PodInjectionPolicyList{}
	for _, item := range obj.(*v1alpha1.PodInjectionPolicyList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested podInjectionPolicies.
func (c *FakePodInjectionPolicies) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(podinjectionpoliciesResource, c.ns, opts))

}

// Patch applies the patch and returns the patched podInjectionPolicy.
func (c *FakePodInjectionPolicies) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1alpha1.PodInjectionPolicy, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(podinjectionpoliciesResource, c.ns, name, data, subresources...), &v1alpha1.PodInjectionPolicy{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.PodInjectionPolicy), err
}
