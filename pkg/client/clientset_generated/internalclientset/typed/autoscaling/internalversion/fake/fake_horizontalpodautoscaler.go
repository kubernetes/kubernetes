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
	autoscaling "k8s.io/kubernetes/pkg/apis/autoscaling"
	core "k8s.io/kubernetes/pkg/client/testing/core"
)

// FakeHorizontalPodAutoscalers implements HorizontalPodAutoscalerInterface
type FakeHorizontalPodAutoscalers struct {
	Fake *FakeAutoscaling
	ns   string
}

var horizontalpodautoscalersResource = schema.GroupVersionResource{Group: "autoscaling", Version: "", Resource: "horizontalpodautoscalers"}

func (c *FakeHorizontalPodAutoscalers) Create(horizontalPodAutoscaler *autoscaling.HorizontalPodAutoscaler) (result *autoscaling.HorizontalPodAutoscaler, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(horizontalpodautoscalersResource, c.ns, horizontalPodAutoscaler), &autoscaling.HorizontalPodAutoscaler{})

	if obj == nil {
		return nil, err
	}
	return obj.(*autoscaling.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalers) Update(horizontalPodAutoscaler *autoscaling.HorizontalPodAutoscaler) (result *autoscaling.HorizontalPodAutoscaler, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(horizontalpodautoscalersResource, c.ns, horizontalPodAutoscaler), &autoscaling.HorizontalPodAutoscaler{})

	if obj == nil {
		return nil, err
	}
	return obj.(*autoscaling.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalers) UpdateStatus(horizontalPodAutoscaler *autoscaling.HorizontalPodAutoscaler) (*autoscaling.HorizontalPodAutoscaler, error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateSubresourceAction(horizontalpodautoscalersResource, "status", c.ns, horizontalPodAutoscaler), &autoscaling.HorizontalPodAutoscaler{})

	if obj == nil {
		return nil, err
	}
	return obj.(*autoscaling.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalers) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(horizontalpodautoscalersResource, c.ns, name), &autoscaling.HorizontalPodAutoscaler{})

	return err
}

func (c *FakeHorizontalPodAutoscalers) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(horizontalpodautoscalersResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &autoscaling.HorizontalPodAutoscalerList{})
	return err
}

func (c *FakeHorizontalPodAutoscalers) Get(name string, options v1.GetOptions) (result *autoscaling.HorizontalPodAutoscaler, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(horizontalpodautoscalersResource, c.ns, name), &autoscaling.HorizontalPodAutoscaler{})

	if obj == nil {
		return nil, err
	}
	return obj.(*autoscaling.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalers) List(opts api.ListOptions) (result *autoscaling.HorizontalPodAutoscalerList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(horizontalpodautoscalersResource, c.ns, opts), &autoscaling.HorizontalPodAutoscalerList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &autoscaling.HorizontalPodAutoscalerList{}
	for _, item := range obj.(*autoscaling.HorizontalPodAutoscalerList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested horizontalPodAutoscalers.
func (c *FakeHorizontalPodAutoscalers) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(horizontalpodautoscalersResource, c.ns, opts))

}

// Patch applies the patch and returns the patched horizontalPodAutoscaler.
func (c *FakeHorizontalPodAutoscalers) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *autoscaling.HorizontalPodAutoscaler, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(horizontalpodautoscalersResource, c.ns, name, data, subresources...), &autoscaling.HorizontalPodAutoscaler{})

	if obj == nil {
		return nil, err
	}
	return obj.(*autoscaling.HorizontalPodAutoscaler), err
}
