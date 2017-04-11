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
	innsmouth "k8s.io/apiserver-builder/example/pkg/apis/innsmouth"
	testing "k8s.io/client-go/testing"
)

// FakeDeepOnes implements DeepOneInterface
type FakeDeepOnes struct {
	Fake *FakeInnsmouth
	ns   string
}

var deeponesResource = schema.GroupVersionResource{Group: "innsmouth", Version: "", Resource: "deepones"}

func (c *FakeDeepOnes) Create(deepOne *innsmouth.DeepOne) (result *innsmouth.DeepOne, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(deeponesResource, c.ns, deepOne), &innsmouth.DeepOne{})

	if obj == nil {
		return nil, err
	}
	return obj.(*innsmouth.DeepOne), err
}

func (c *FakeDeepOnes) Update(deepOne *innsmouth.DeepOne) (result *innsmouth.DeepOne, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(deeponesResource, c.ns, deepOne), &innsmouth.DeepOne{})

	if obj == nil {
		return nil, err
	}
	return obj.(*innsmouth.DeepOne), err
}

func (c *FakeDeepOnes) UpdateStatus(deepOne *innsmouth.DeepOne) (*innsmouth.DeepOne, error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceAction(deeponesResource, "status", c.ns, deepOne), &innsmouth.DeepOne{})

	if obj == nil {
		return nil, err
	}
	return obj.(*innsmouth.DeepOne), err
}

func (c *FakeDeepOnes) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(deeponesResource, c.ns, name), &innsmouth.DeepOne{})

	return err
}

func (c *FakeDeepOnes) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(deeponesResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &innsmouth.DeepOneList{})
	return err
}

func (c *FakeDeepOnes) Get(name string, options v1.GetOptions) (result *innsmouth.DeepOne, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(deeponesResource, c.ns, name), &innsmouth.DeepOne{})

	if obj == nil {
		return nil, err
	}
	return obj.(*innsmouth.DeepOne), err
}

func (c *FakeDeepOnes) List(opts v1.ListOptions) (result *innsmouth.DeepOneList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(deeponesResource, c.ns, opts), &innsmouth.DeepOneList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &innsmouth.DeepOneList{}
	for _, item := range obj.(*innsmouth.DeepOneList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested deepOnes.
func (c *FakeDeepOnes) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(deeponesResource, c.ns, opts))

}

// Patch applies the patch and returns the patched deepOne.
func (c *FakeDeepOnes) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *innsmouth.DeepOne, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(deeponesResource, c.ns, name, data, subresources...), &innsmouth.DeepOne{})

	if obj == nil {
		return nil, err
	}
	return obj.(*innsmouth.DeepOne), err
}
