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
	v1 "k8s.io/apiserver-builder/example/pkg/apis/innsmouth/v1"
	testing "k8s.io/client-go/testing"
)

// FakeDeepOnes implements DeepOneInterface
type FakeDeepOnes struct {
	Fake *FakeInnsmouthV1
	ns   string
}

var deeponesResource = schema.GroupVersionResource{Group: "innsmouth.k8s.io", Version: "v1", Resource: "deepones"}

func (c *FakeDeepOnes) Create(deepOne *v1.DeepOne) (result *v1.DeepOne, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(deeponesResource, c.ns, deepOne), &v1.DeepOne{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.DeepOne), err
}

func (c *FakeDeepOnes) Update(deepOne *v1.DeepOne) (result *v1.DeepOne, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(deeponesResource, c.ns, deepOne), &v1.DeepOne{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.DeepOne), err
}

func (c *FakeDeepOnes) UpdateStatus(deepOne *v1.DeepOne) (*v1.DeepOne, error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceAction(deeponesResource, "status", c.ns, deepOne), &v1.DeepOne{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.DeepOne), err
}

func (c *FakeDeepOnes) Delete(name string, options *meta_v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(deeponesResource, c.ns, name), &v1.DeepOne{})

	return err
}

func (c *FakeDeepOnes) DeleteCollection(options *meta_v1.DeleteOptions, listOptions meta_v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(deeponesResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1.DeepOneList{})
	return err
}

func (c *FakeDeepOnes) Get(name string, options meta_v1.GetOptions) (result *v1.DeepOne, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(deeponesResource, c.ns, name), &v1.DeepOne{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.DeepOne), err
}

func (c *FakeDeepOnes) List(opts meta_v1.ListOptions) (result *v1.DeepOneList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(deeponesResource, c.ns, opts), &v1.DeepOneList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1.DeepOneList{}
	for _, item := range obj.(*v1.DeepOneList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested deepOnes.
func (c *FakeDeepOnes) Watch(opts meta_v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(deeponesResource, c.ns, opts))

}

// Patch applies the patch and returns the patched deepOne.
func (c *FakeDeepOnes) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1.DeepOne, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(deeponesResource, c.ns, name, data, subresources...), &v1.DeepOne{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.DeepOne), err
}
