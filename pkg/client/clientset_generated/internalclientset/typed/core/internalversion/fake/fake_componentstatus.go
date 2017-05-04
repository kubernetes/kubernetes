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
	api "k8s.io/kubernetes/pkg/api"
)

// FakeComponentStatuses implements ComponentStatusInterface
type FakeComponentStatuses struct {
	Fake *FakeCore
}

var componentstatusesResource = schema.GroupVersionResource{Group: "", Version: "", Resource: "componentstatuses"}

var componentstatusesKind = schema.GroupVersionKind{Group: "", Version: "", Kind: "ComponentStatus"}

func (c *FakeComponentStatuses) Create(componentStatus *api.ComponentStatus) (result *api.ComponentStatus, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(componentstatusesResource, componentStatus), &api.ComponentStatus{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.ComponentStatus), err
}

func (c *FakeComponentStatuses) Update(componentStatus *api.ComponentStatus) (result *api.ComponentStatus, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(componentstatusesResource, componentStatus), &api.ComponentStatus{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.ComponentStatus), err
}

func (c *FakeComponentStatuses) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(componentstatusesResource, name), &api.ComponentStatus{})
	return err
}

func (c *FakeComponentStatuses) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(componentstatusesResource, listOptions)

	_, err := c.Fake.Invokes(action, &api.ComponentStatusList{})
	return err
}

func (c *FakeComponentStatuses) Get(name string, options v1.GetOptions) (result *api.ComponentStatus, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(componentstatusesResource, name), &api.ComponentStatus{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.ComponentStatus), err
}

func (c *FakeComponentStatuses) List(opts v1.ListOptions) (result *api.ComponentStatusList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(componentstatusesResource, componentstatusesKind, opts), &api.ComponentStatusList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &api.ComponentStatusList{}
	for _, item := range obj.(*api.ComponentStatusList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested componentStatuses.
func (c *FakeComponentStatuses) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(componentstatusesResource, opts))
}

// Patch applies the patch and returns the patched componentStatus.
func (c *FakeComponentStatuses) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *api.ComponentStatus, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(componentstatusesResource, name, data, subresources...), &api.ComponentStatus{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.ComponentStatus), err
}
