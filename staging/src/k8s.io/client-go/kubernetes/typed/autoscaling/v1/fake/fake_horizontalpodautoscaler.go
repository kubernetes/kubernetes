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
	api "k8s.io/client-go/pkg/api"
	unversioned "k8s.io/client-go/pkg/api/unversioned"
	api_v1 "k8s.io/client-go/pkg/api/v1"
	v1 "k8s.io/client-go/pkg/apis/autoscaling/v1"
	labels "k8s.io/client-go/pkg/labels"
	watch "k8s.io/client-go/pkg/watch"
	testing "k8s.io/client-go/testing"
)

// FakeHorizontalPodAutoscalers implements HorizontalPodAutoscalerInterface
type FakeHorizontalPodAutoscalers struct {
	Fake *FakeAutoscalingV1
	ns   string
}

var horizontalpodautoscalersResource = unversioned.GroupVersionResource{Group: "autoscaling", Version: "v1", Resource: "horizontalpodautoscalers"}

func (c *FakeHorizontalPodAutoscalers) Create(horizontalPodAutoscaler *v1.HorizontalPodAutoscaler) (result *v1.HorizontalPodAutoscaler, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(horizontalpodautoscalersResource, c.ns, horizontalPodAutoscaler), &v1.HorizontalPodAutoscaler{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalers) Update(horizontalPodAutoscaler *v1.HorizontalPodAutoscaler) (result *v1.HorizontalPodAutoscaler, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(horizontalpodautoscalersResource, c.ns, horizontalPodAutoscaler), &v1.HorizontalPodAutoscaler{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalers) UpdateStatus(horizontalPodAutoscaler *v1.HorizontalPodAutoscaler) (*v1.HorizontalPodAutoscaler, error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceAction(horizontalpodautoscalersResource, "status", c.ns, horizontalPodAutoscaler), &v1.HorizontalPodAutoscaler{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalers) Delete(name string, options *api_v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(horizontalpodautoscalersResource, c.ns, name), &v1.HorizontalPodAutoscaler{})

	return err
}

func (c *FakeHorizontalPodAutoscalers) DeleteCollection(options *api_v1.DeleteOptions, listOptions api_v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(horizontalpodautoscalersResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1.HorizontalPodAutoscalerList{})
	return err
}

func (c *FakeHorizontalPodAutoscalers) Get(name string) (result *v1.HorizontalPodAutoscaler, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(horizontalpodautoscalersResource, c.ns, name), &v1.HorizontalPodAutoscaler{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.HorizontalPodAutoscaler), err
}

func (c *FakeHorizontalPodAutoscalers) List(opts api_v1.ListOptions) (result *v1.HorizontalPodAutoscalerList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(horizontalpodautoscalersResource, c.ns, opts), &v1.HorizontalPodAutoscalerList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1.HorizontalPodAutoscalerList{}
	for _, item := range obj.(*v1.HorizontalPodAutoscalerList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested horizontalPodAutoscalers.
func (c *FakeHorizontalPodAutoscalers) Watch(opts api_v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(horizontalpodautoscalersResource, c.ns, opts))

}

// Patch applies the patch and returns the patched horizontalPodAutoscaler.
func (c *FakeHorizontalPodAutoscalers) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.HorizontalPodAutoscaler, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(horizontalpodautoscalersResource, c.ns, name, data, subresources...), &v1.HorizontalPodAutoscaler{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.HorizontalPodAutoscaler), err
}
