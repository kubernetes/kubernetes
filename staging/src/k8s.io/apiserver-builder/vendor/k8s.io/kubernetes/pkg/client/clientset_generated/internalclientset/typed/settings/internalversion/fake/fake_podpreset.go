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
	settings "k8s.io/kubernetes/pkg/apis/settings"
)

// FakePodPresets implements PodPresetInterface
type FakePodPresets struct {
	Fake *FakeSettings
	ns   string
}

var podpresetsResource = schema.GroupVersionResource{Group: "settings.k8s.io", Version: "", Resource: "podpresets"}

func (c *FakePodPresets) Create(podPreset *settings.PodPreset) (result *settings.PodPreset, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(podpresetsResource, c.ns, podPreset), &settings.PodPreset{})

	if obj == nil {
		return nil, err
	}
	return obj.(*settings.PodPreset), err
}

func (c *FakePodPresets) Update(podPreset *settings.PodPreset) (result *settings.PodPreset, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(podpresetsResource, c.ns, podPreset), &settings.PodPreset{})

	if obj == nil {
		return nil, err
	}
	return obj.(*settings.PodPreset), err
}

func (c *FakePodPresets) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(podpresetsResource, c.ns, name), &settings.PodPreset{})

	return err
}

func (c *FakePodPresets) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(podpresetsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &settings.PodPresetList{})
	return err
}

func (c *FakePodPresets) Get(name string, options v1.GetOptions) (result *settings.PodPreset, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(podpresetsResource, c.ns, name), &settings.PodPreset{})

	if obj == nil {
		return nil, err
	}
	return obj.(*settings.PodPreset), err
}

func (c *FakePodPresets) List(opts v1.ListOptions) (result *settings.PodPresetList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(podpresetsResource, c.ns, opts), &settings.PodPresetList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &settings.PodPresetList{}
	for _, item := range obj.(*settings.PodPresetList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested podPresets.
func (c *FakePodPresets) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(podpresetsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched podPreset.
func (c *FakePodPresets) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *settings.PodPreset, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(podpresetsResource, c.ns, name, data, subresources...), &settings.PodPreset{})

	if obj == nil {
		return nil, err
	}
	return obj.(*settings.PodPreset), err
}
