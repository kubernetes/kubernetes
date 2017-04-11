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
	v1beta1 "k8s.io/apiserver-builder/example/pkg/apis/miskatonic/v1beta1"
	testing "k8s.io/client-go/testing"
)

// FakeScales implements ScaleInterface
type FakeScales struct {
	Fake *FakeMiskatonicV1beta1
	ns   string
}

var scalesResource = schema.GroupVersionResource{Group: "miskatonic.k8s.io", Version: "v1beta1", Resource: "scales"}

func (c *FakeScales) Create(scale *v1beta1.Scale) (result *v1beta1.Scale, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(scalesResource, c.ns, scale), &v1beta1.Scale{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Scale), err
}

func (c *FakeScales) Update(scale *v1beta1.Scale) (result *v1beta1.Scale, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(scalesResource, c.ns, scale), &v1beta1.Scale{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Scale), err
}

func (c *FakeScales) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(scalesResource, c.ns, name), &v1beta1.Scale{})

	return err
}

func (c *FakeScales) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(scalesResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1beta1.ScaleList{})
	return err
}

func (c *FakeScales) Get(name string, options v1.GetOptions) (result *v1beta1.Scale, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(scalesResource, c.ns, name), &v1beta1.Scale{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Scale), err
}

func (c *FakeScales) List(opts v1.ListOptions) (result *v1beta1.ScaleList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(scalesResource, c.ns, opts), &v1beta1.ScaleList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1beta1.ScaleList{}
	for _, item := range obj.(*v1beta1.ScaleList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested scales.
func (c *FakeScales) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(scalesResource, c.ns, opts))

}

// Patch applies the patch and returns the patched scale.
func (c *FakeScales) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1beta1.Scale, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(scalesResource, c.ns, name, data, subresources...), &v1beta1.Scale{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1beta1.Scale), err
}
