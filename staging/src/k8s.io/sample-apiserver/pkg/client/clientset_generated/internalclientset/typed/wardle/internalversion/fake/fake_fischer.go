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
	testing "k8s.io/client-go/testing"
	wardle "k8s.io/sample-apiserver/pkg/apis/wardle"
)

// FakeFischers implements FischerInterface
type FakeFischers struct {
	Fake *FakeWardle
}

var fischersResource = schema.GroupVersionResource{Group: "wardle.k8s.io", Version: "", Resource: "fischers"}

var fischersKind = schema.GroupVersionKind{Group: "wardle.k8s.io", Version: "", Kind: "Fischer"}

func (c *FakeFischers) Create(fischer *wardle.Fischer) (result *wardle.Fischer, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(fischersResource, fischer), &wardle.Fischer{})
	if obj == nil {
		return nil, err
	}
	return obj.(*wardle.Fischer), err
}

func (c *FakeFischers) Update(fischer *wardle.Fischer) (result *wardle.Fischer, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(fischersResource, fischer), &wardle.Fischer{})
	if obj == nil {
		return nil, err
	}
	return obj.(*wardle.Fischer), err
}

func (c *FakeFischers) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(fischersResource, name), &wardle.Fischer{})
	return err
}

func (c *FakeFischers) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(fischersResource, listOptions)

	_, err := c.Fake.Invokes(action, &wardle.FischerList{})
	return err
}

func (c *FakeFischers) Get(name string, options v1.GetOptions) (result *wardle.Fischer, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(fischersResource, name), &wardle.Fischer{})
	if obj == nil {
		return nil, err
	}
	return obj.(*wardle.Fischer), err
}

func (c *FakeFischers) List(opts v1.ListOptions) (result *wardle.FischerList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(fischersResource, fischersKind, opts), &wardle.FischerList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &wardle.FischerList{}
	for _, item := range obj.(*wardle.FischerList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested fischers.
func (c *FakeFischers) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(fischersResource, opts))
}

// Patch applies the patch and returns the patched fischer.
func (c *FakeFischers) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *wardle.Fischer, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(fischersResource, name, data, subresources...), &wardle.Fischer{})
	if obj == nil {
		return nil, err
	}
	return obj.(*wardle.Fischer), err
}
