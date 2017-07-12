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

// FakeFlunders implements FlunderInterface
type FakeFlunders struct {
	Fake *FakeWardle
	ns   string
}

var flundersResource = schema.GroupVersionResource{Group: "wardle.k8s.io", Version: "", Resource: "flunders"}

var flundersKind = schema.GroupVersionKind{Group: "wardle.k8s.io", Version: "", Kind: "Flunder"}

func (c *FakeFlunders) Create(flunder *wardle.Flunder) (result *wardle.Flunder, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(flundersResource, c.ns, flunder), &wardle.Flunder{})

	if obj == nil {
		return nil, err
	}
	return obj.(*wardle.Flunder), err
}

func (c *FakeFlunders) Update(flunder *wardle.Flunder) (result *wardle.Flunder, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(flundersResource, c.ns, flunder), &wardle.Flunder{})

	if obj == nil {
		return nil, err
	}
	return obj.(*wardle.Flunder), err
}

func (c *FakeFlunders) UpdateStatus(flunder *wardle.Flunder) (*wardle.Flunder, error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceAction(flundersResource, "status", c.ns, flunder), &wardle.Flunder{})

	if obj == nil {
		return nil, err
	}
	return obj.(*wardle.Flunder), err
}

func (c *FakeFlunders) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(flundersResource, c.ns, name), &wardle.Flunder{})

	return err
}

func (c *FakeFlunders) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(flundersResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &wardle.FlunderList{})
	return err
}

func (c *FakeFlunders) Get(name string, options v1.GetOptions) (result *wardle.Flunder, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(flundersResource, c.ns, name), &wardle.Flunder{})

	if obj == nil {
		return nil, err
	}
	return obj.(*wardle.Flunder), err
}

func (c *FakeFlunders) List(opts v1.ListOptions) (result *wardle.FlunderList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(flundersResource, flundersKind, c.ns, opts), &wardle.FlunderList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &wardle.FlunderList{}
	for _, item := range obj.(*wardle.FlunderList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested flunders.
func (c *FakeFlunders) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(flundersResource, c.ns, opts))

}

// Patch applies the patch and returns the patched flunder.
func (c *FakeFlunders) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *wardle.Flunder, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(flundersResource, c.ns, name, data, subresources...), &wardle.Flunder{})

	if obj == nil {
		return nil, err
	}
	return obj.(*wardle.Flunder), err
}
