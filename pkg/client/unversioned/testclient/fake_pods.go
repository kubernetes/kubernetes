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

// FakePods implements PodsInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakePods struct {
	Fake      *Fake
	Namespace string
}

func (c *FakePods) Get(name string) (*api.Pod, error) {
	obj, err := c.Fake.Invokes(NewGetAction("pods", c.Namespace, name), &api.Pod{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Pod), err
}

func (c *FakePods) List(label labels.Selector, field fields.Selector) (*api.PodList, error) {
	obj, err := c.Fake.Invokes(NewListAction("pods", c.Namespace, label, field), &api.PodList{})
	if obj == nil {
		return nil, err
	}
	list := &api.PodList{}
	for _, pod := range obj.(*api.PodList).Items {
		if label.Matches(labels.Set(pod.Labels)) {
			list.Items = append(list.Items, pod)
		}
	}
	return list, err
}

func (c *FakePods) Create(pod *api.Pod) (*api.Pod, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("pods", c.Namespace, pod), pod)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Pod), err
}

func (c *FakePods) Update(pod *api.Pod) (*api.Pod, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("pods", c.Namespace, pod), pod)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Pod), err
}

func (c *FakePods) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewDeleteAction("pods", c.Namespace, name), &api.Pod{})
	return err
}

func (c *FakePods) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Invokes(NewWatchAction("pods", c.Namespace, label, field, resourceVersion), nil)
	return c.Fake.Watch, c.Fake.Err()
}

func (c *FakePods) Bind(binding *api.Binding) error {
	action := CreateActionImpl{}
	action.Verb = "create"
	action.Resource = "pods"
	action.Subresource = "bindings"
	action.Object = binding

	_, err := c.Fake.Invokes(action, binding)
	return err
}

func (c *FakePods) UpdateStatus(pod *api.Pod) (*api.Pod, error) {
	action := UpdateActionImpl{}
	action.Verb = "update"
	action.Resource = "pods"
	action.Subresource = "status"
	action.Object = pod

	obj, err := c.Fake.Invokes(action, pod)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Pod), err
}
