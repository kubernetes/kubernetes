/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeHorizontalPodAutoscalers implements HorizontalPodAutoscalerInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakeHorizontalPodAutoscalers struct {
	Fake      *FakeExperimental
	Namespace string
}

func (c *FakeHorizontalPodAutoscalers) Get(name string) (*expapi.HorizontalPodAutoscaler, error) {
	obj, err := c.Fake.Invokes(NewGetAction("horizontalpodautoscalers", c.Namespace, name), &expapi.HorizontalPodAutoscaler{})
	if obj == nil {
		return nil, err
	}

	return obj.(*expapi.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalers) List(label labels.Selector, field fields.Selector) (*expapi.HorizontalPodAutoscalerList, error) {
	obj, err := c.Fake.Invokes(NewListAction("horizontalpodautoscalers", c.Namespace, label, field), &expapi.HorizontalPodAutoscalerList{})
	if obj == nil {
		return nil, err
	}
	list := &expapi.HorizontalPodAutoscalerList{}
	for _, a := range obj.(*expapi.HorizontalPodAutoscalerList).Items {
		if label.Matches(labels.Set(a.Labels)) {
			list.Items = append(list.Items, a)
		}
	}
	return list, err
}

func (c *FakeHorizontalPodAutoscalers) Create(a *expapi.HorizontalPodAutoscaler) (*expapi.HorizontalPodAutoscaler, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("horizontalpodautoscalers", c.Namespace, a), a)
	if obj == nil {
		return nil, err
	}

	return obj.(*expapi.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalers) Update(a *expapi.HorizontalPodAutoscaler) (*expapi.HorizontalPodAutoscaler, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("horizontalpodautoscalers", c.Namespace, a), a)
	if obj == nil {
		return nil, err
	}

	return obj.(*expapi.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalers) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewDeleteAction("horizontalpodautoscalers", c.Namespace, name), &expapi.HorizontalPodAutoscaler{})
	return err
}

func (c *FakeHorizontalPodAutoscalers) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("horizontalpodautoscalers", c.Namespace, label, field, resourceVersion))
}
