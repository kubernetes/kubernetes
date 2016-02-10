/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeHorizontalPodAutoscalers implements HorizontalPodAutoscalerInterface
type FakeHorizontalPodAutoscalers struct {
	Fake *FakeExtensions
	ns   string
}

func (c *FakeHorizontalPodAutoscalers) Create(horizontalPodAutoscaler *extensions.HorizontalPodAutoscaler) (result *extensions.HorizontalPodAutoscaler, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction("horizontalpodautoscalers", c.ns, horizontalPodAutoscaler), &extensions.HorizontalPodAutoscaler{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalers) Update(horizontalPodAutoscaler *extensions.HorizontalPodAutoscaler) (result *extensions.HorizontalPodAutoscaler, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction("horizontalpodautoscalers", c.ns, horizontalPodAutoscaler), &extensions.HorizontalPodAutoscaler{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalers) UpdateStatus(horizontalPodAutoscaler *extensions.HorizontalPodAutoscaler) (*extensions.HorizontalPodAutoscaler, error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateSubresourceAction("horizontalpodautoscalers", "status", c.ns, horizontalPodAutoscaler), &extensions.HorizontalPodAutoscaler{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalers) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction("horizontalpodautoscalers", c.ns, name), &extensions.HorizontalPodAutoscaler{})

	return err
}

func (c *FakeHorizontalPodAutoscalers) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction("horizontalpodautoscalers", c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &extensions.HorizontalPodAutoscalerList{})
	return err
}

func (c *FakeHorizontalPodAutoscalers) Get(name string) (result *extensions.HorizontalPodAutoscaler, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction("horizontalpodautoscalers", c.ns, name), &extensions.HorizontalPodAutoscaler{})

	if obj == nil {
		return nil, err
	}
	return obj.(*extensions.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalers) List(opts api.ListOptions) (result *extensions.HorizontalPodAutoscalerList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction("horizontalpodautoscalers", c.ns, opts), &extensions.HorizontalPodAutoscalerList{})

	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
	if label == nil {
		label = labels.Everything()
	}
	list := &extensions.HorizontalPodAutoscalerList{}
	for _, item := range obj.(*extensions.HorizontalPodAutoscalerList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested horizontalPodAutoscalers.
func (c *FakeHorizontalPodAutoscalers) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction("horizontalpodautoscalers", c.ns, opts))

}
