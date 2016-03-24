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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeHorizontalPodAutoscalers implements HorizontalPodAutoscalerInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
type FakeHorizontalPodAutoscalers struct {
	Fake      *FakeExperimental
	Namespace string
}

func (c *FakeHorizontalPodAutoscalers) Get(name string) (*extensions.HorizontalPodAutoscaler, error) {
	obj, err := c.Fake.Invokes(NewGetAction("horizontalpodautoscalers", c.Namespace, name), &extensions.HorizontalPodAutoscaler{})
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalers) List(opts api.ListOptions) (*extensions.HorizontalPodAutoscalerList, error) {
	obj, err := c.Fake.Invokes(NewListAction("horizontalpodautoscalers", c.Namespace, opts), &extensions.HorizontalPodAutoscalerList{})
	if obj == nil {
		return nil, err
	}
	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &extensions.HorizontalPodAutoscalerList{}
	for _, a := range obj.(*extensions.HorizontalPodAutoscalerList).Items {
		if label.Matches(labels.Set(a.Labels)) {
			list.Items = append(list.Items, a)
		}
	}
	return list, err
}

func (c *FakeHorizontalPodAutoscalers) Create(a *extensions.HorizontalPodAutoscaler) (*extensions.HorizontalPodAutoscaler, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("horizontalpodautoscalers", c.Namespace, a), a)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalers) Update(a *extensions.HorizontalPodAutoscaler) (*extensions.HorizontalPodAutoscaler, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("horizontalpodautoscalers", c.Namespace, a), a)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalers) UpdateStatus(a *extensions.HorizontalPodAutoscaler) (*extensions.HorizontalPodAutoscaler, error) {
	obj, err := c.Fake.Invokes(NewUpdateSubresourceAction("horizontalpodautoscalers", "status", c.Namespace, a), &extensions.HorizontalPodAutoscaler{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalers) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewDeleteAction("horizontalpodautoscalers", c.Namespace, name), &extensions.HorizontalPodAutoscaler{})
	return err
}

func (c *FakeHorizontalPodAutoscalers) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("horizontalpodautoscalers", c.Namespace, opts))
}

// FakeHorizontalPodAutoscalers implements HorizontalPodAutoscalerInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the methods you want to test easier.
// This is a test implementation of HorizontalPodAutoscalersV1
// TODO(piosz): get back to one client implementation once HPA will be graduated to GA completely
type FakeHorizontalPodAutoscalersV1 struct {
	Fake      *FakeAutoscaling
	Namespace string
}

func (c *FakeHorizontalPodAutoscalersV1) Get(name string) (*extensions.HorizontalPodAutoscaler, error) {
	obj, err := c.Fake.Invokes(NewGetAction("horizontalpodautoscalers", c.Namespace, name), &extensions.HorizontalPodAutoscaler{})
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalersV1) List(opts api.ListOptions) (*extensions.HorizontalPodAutoscalerList, error) {
	obj, err := c.Fake.Invokes(NewListAction("horizontalpodautoscalers", c.Namespace, opts), &extensions.HorizontalPodAutoscalerList{})
	if obj == nil {
		return nil, err
	}
	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &extensions.HorizontalPodAutoscalerList{}
	for _, a := range obj.(*extensions.HorizontalPodAutoscalerList).Items {
		if label.Matches(labels.Set(a.Labels)) {
			list.Items = append(list.Items, a)
		}
	}
	return list, err
}

func (c *FakeHorizontalPodAutoscalersV1) Create(a *extensions.HorizontalPodAutoscaler) (*extensions.HorizontalPodAutoscaler, error) {
	obj, err := c.Fake.Invokes(NewCreateAction("horizontalpodautoscalers", c.Namespace, a), a)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalersV1) Update(a *extensions.HorizontalPodAutoscaler) (*extensions.HorizontalPodAutoscaler, error) {
	obj, err := c.Fake.Invokes(NewUpdateAction("horizontalpodautoscalers", c.Namespace, a), a)
	if obj == nil {
		return nil, err
	}

	return obj.(*extensions.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalersV1) UpdateStatus(a *extensions.HorizontalPodAutoscaler) (*extensions.HorizontalPodAutoscaler, error) {
	obj, err := c.Fake.Invokes(NewUpdateSubresourceAction("horizontalpodautoscalers", "status", c.Namespace, a), &extensions.HorizontalPodAutoscaler{})
	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalersV1) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewDeleteAction("horizontalpodautoscalers", c.Namespace, name), &extensions.HorizontalPodAutoscaler{})
	return err
}

func (c *FakeHorizontalPodAutoscalersV1) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewWatchAction("horizontalpodautoscalers", c.Namespace, opts))
}
