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
	watch "k8s.io/apimachinery/pkg/watch"
	api "k8s.io/kubernetes/pkg/api"
	policy "k8s.io/kubernetes/pkg/apis/policy"
	core "k8s.io/kubernetes/pkg/client/testing/core"
)

// FakePodDisruptionBudgets implements PodDisruptionBudgetInterface
type FakePodDisruptionBudgets struct {
	Fake *FakePolicy
	ns   string
}

var poddisruptionbudgetsResource = schema.GroupVersionResource{Group: "policy", Version: "", Resource: "poddisruptionbudgets"}

func (c *FakePodDisruptionBudgets) Create(podDisruptionBudget *policy.PodDisruptionBudget) (result *policy.PodDisruptionBudget, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(poddisruptionbudgetsResource, c.ns, podDisruptionBudget), &policy.PodDisruptionBudget{})

	if obj == nil {
		return nil, err
	}
	return obj.(*policy.PodDisruptionBudget), err
}

func (c *FakePodDisruptionBudgets) Update(podDisruptionBudget *policy.PodDisruptionBudget) (result *policy.PodDisruptionBudget, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(poddisruptionbudgetsResource, c.ns, podDisruptionBudget), &policy.PodDisruptionBudget{})

	if obj == nil {
		return nil, err
	}
	return obj.(*policy.PodDisruptionBudget), err
}

func (c *FakePodDisruptionBudgets) UpdateStatus(podDisruptionBudget *policy.PodDisruptionBudget) (*policy.PodDisruptionBudget, error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateSubresourceAction(poddisruptionbudgetsResource, "status", c.ns, podDisruptionBudget), &policy.PodDisruptionBudget{})

	if obj == nil {
		return nil, err
	}
	return obj.(*policy.PodDisruptionBudget), err
}

func (c *FakePodDisruptionBudgets) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(poddisruptionbudgetsResource, c.ns, name), &policy.PodDisruptionBudget{})

	return err
}

func (c *FakePodDisruptionBudgets) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(poddisruptionbudgetsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &policy.PodDisruptionBudgetList{})
	return err
}

func (c *FakePodDisruptionBudgets) Get(name string, options v1.GetOptions) (result *policy.PodDisruptionBudget, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(poddisruptionbudgetsResource, c.ns, name), &policy.PodDisruptionBudget{})

	if obj == nil {
		return nil, err
	}
	return obj.(*policy.PodDisruptionBudget), err
}

func (c *FakePodDisruptionBudgets) List(opts api.ListOptions) (result *policy.PodDisruptionBudgetList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(poddisruptionbudgetsResource, c.ns, opts), &policy.PodDisruptionBudgetList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &policy.PodDisruptionBudgetList{}
	for _, item := range obj.(*policy.PodDisruptionBudgetList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested podDisruptionBudgets.
func (c *FakePodDisruptionBudgets) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(poddisruptionbudgetsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched podDisruptionBudget.
func (c *FakePodDisruptionBudgets) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *policy.PodDisruptionBudget, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(poddisruptionbudgetsResource, c.ns, name, data, subresources...), &policy.PodDisruptionBudget{})

	if obj == nil {
		return nil, err
	}
	return obj.(*policy.PodDisruptionBudget), err
}
