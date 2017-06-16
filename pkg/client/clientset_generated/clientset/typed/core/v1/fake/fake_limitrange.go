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
	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	testing "k8s.io/client-go/testing"
	v1 "k8s.io/kubernetes/pkg/api/v1"
)

// FakeLimitRanges implements LimitRangeInterface
type FakeLimitRanges struct {
	Fake *FakeCoreV1
	ns   string
}

var limitrangesResource = schema.GroupVersionResource{Group: "", Version: "v1", Resource: "limitranges"}

var limitrangesKind = schema.GroupVersionKind{Group: "", Version: "v1", Kind: "LimitRange"}

func (c *FakeLimitRanges) Create(limitRange *v1.LimitRange) (result *v1.LimitRange, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(limitrangesResource, c.ns, limitRange), &v1.LimitRange{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.LimitRange), err
}

func (c *FakeLimitRanges) Update(limitRange *v1.LimitRange) (result *v1.LimitRange, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(limitrangesResource, c.ns, limitRange), &v1.LimitRange{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.LimitRange), err
}

func (c *FakeLimitRanges) Delete(name string, options *meta_v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(limitrangesResource, c.ns, name), &v1.LimitRange{})

	return err
}

func (c *FakeLimitRanges) DeleteCollection(options *meta_v1.DeleteOptions, listOptions meta_v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(limitrangesResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1.LimitRangeList{})
	return err
}

func (c *FakeLimitRanges) Get(name string, options meta_v1.GetOptions) (result *v1.LimitRange, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(limitrangesResource, c.ns, name), &v1.LimitRange{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.LimitRange), err
}

func (c *FakeLimitRanges) List(opts meta_v1.ListOptions) (result *v1.LimitRangeList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(limitrangesResource, limitrangesKind, c.ns, opts), &v1.LimitRangeList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1.LimitRangeList{}
	for _, item := range obj.(*v1.LimitRangeList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested limitRanges.
func (c *FakeLimitRanges) Watch(opts meta_v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(limitrangesResource, c.ns, opts))

}

// Patch applies the patch and returns the patched limitRange.
func (c *FakeLimitRanges) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1.LimitRange, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(limitrangesResource, c.ns, name, data, subresources...), &v1.LimitRange{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.LimitRange), err
}
