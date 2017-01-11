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
	core "k8s.io/kubernetes/pkg/client/testing/core"
)

// FakeLimitRanges implements LimitRangeInterface
type FakeLimitRanges struct {
	Fake *FakeCore
	ns   string
}

var limitrangesResource = schema.GroupVersionResource{Group: "", Version: "", Resource: "limitranges"}

func (c *FakeLimitRanges) Create(limitRange *api.LimitRange) (result *api.LimitRange, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(limitrangesResource, c.ns, limitRange), &api.LimitRange{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.LimitRange), err
}

func (c *FakeLimitRanges) Update(limitRange *api.LimitRange) (result *api.LimitRange, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(limitrangesResource, c.ns, limitRange), &api.LimitRange{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.LimitRange), err
}

func (c *FakeLimitRanges) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(limitrangesResource, c.ns, name), &api.LimitRange{})

	return err
}

func (c *FakeLimitRanges) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewDeleteCollectionAction(limitrangesResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &api.LimitRangeList{})
	return err
}

func (c *FakeLimitRanges) Get(name string, options v1.GetOptions) (result *api.LimitRange, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(limitrangesResource, c.ns, name), &api.LimitRange{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.LimitRange), err
}

func (c *FakeLimitRanges) List(opts api.ListOptions) (result *api.LimitRangeList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(limitrangesResource, c.ns, opts), &api.LimitRangeList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &api.LimitRangeList{}
	for _, item := range obj.(*api.LimitRangeList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested limitRanges.
func (c *FakeLimitRanges) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(limitrangesResource, c.ns, opts))

}

// Patch applies the patch and returns the patched limitRange.
func (c *FakeLimitRanges) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *api.LimitRange, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(limitrangesResource, c.ns, name, data, subresources...), &api.LimitRange{})

	if obj == nil {
		return nil, err
	}
	return obj.(*api.LimitRange), err
}
