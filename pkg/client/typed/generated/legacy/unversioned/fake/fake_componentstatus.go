/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	api "k8s.io/kubernetes/pkg/api"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeComponentStatus implements ComponentStatusInterface
type FakeComponentStatus struct {
	Fake *FakeLegacy
}

func (c *FakeComponentStatus) Create(componentStatus *api.ComponentStatus) (result *api.ComponentStatus, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootCreateAction("componentStatus", componentStatus), &api.ComponentStatus{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.ComponentStatus), err
}

func (c *FakeComponentStatus) Update(componentStatus *api.ComponentStatus) (result *api.ComponentStatus, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateAction("componentStatus", componentStatus), &api.ComponentStatus{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.ComponentStatus), err
}

func (c *FakeComponentStatus) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewRootDeleteAction("componentStatus", name), &api.ComponentStatus{})
	return err
}

func (c *FakeComponentStatus) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewRootDeleteCollectionAction("events", listOptions)

	_, err := c.Fake.Invokes(action, &api.ComponentStatusList{})
	return err
}

func (c *FakeComponentStatus) Get(name string) (result *api.ComponentStatus, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootGetAction("componentStatus", name), &api.ComponentStatus{})
	if obj == nil {
		return nil, err
	}
	return obj.(*api.ComponentStatus), err
}

func (c *FakeComponentStatus) List(opts api.ListOptions) (result *api.ComponentStatusList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootListAction("componentStatus", opts), &api.ComponentStatusList{})
	if obj == nil {
		return nil, err
	}

	label := opts.LabelSelector
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

// Watch returns a watch.Interface that watches the requested componentStatus.
func (c *FakeComponentStatus) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewRootWatchAction("componentStatus", opts))
}
