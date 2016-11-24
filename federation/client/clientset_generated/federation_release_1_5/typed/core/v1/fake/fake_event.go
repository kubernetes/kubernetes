/*
Copyright 2016 The Kubernetes Authors.

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
	v1 "k8s.io/kubernetes/pkg/api/v1"
	core "k8s.io/kubernetes/pkg/client/testing/core"
	labels "k8s.io/kubernetes/pkg/labels"
	schema "k8s.io/kubernetes/pkg/runtime/schema"
	watch "k8s.io/kubernetes/pkg/watch"
)

// FakeEvents implements EventInterface
type FakeEvents struct {
	Fake *FakeCoreV1
	ns   string
}

var eventsResource = schema.GroupVersionResource{Group: "", Version: "v1", Resource: "events"}

func (c *FakeEvents) Create(event *v1.Event) (result *v1.Event, err error) {
	obj, err := c.Fake.
		Invokes(core.NewCreateAction(eventsResource, c.ns, event), &v1.Event{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Event), err
}

func (c *FakeEvents) Update(event *v1.Event) (result *v1.Event, err error) {
	obj, err := c.Fake.
		Invokes(core.NewUpdateAction(eventsResource, c.ns, event), &v1.Event{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Event), err
}

func (c *FakeEvents) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewDeleteAction(eventsResource, c.ns, name), &v1.Event{})

	return err
}

func (c *FakeEvents) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := core.NewDeleteCollectionAction(eventsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1.EventList{})
	return err
}

func (c *FakeEvents) Get(name string) (result *v1.Event, err error) {
	obj, err := c.Fake.
		Invokes(core.NewGetAction(eventsResource, c.ns, name), &v1.Event{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Event), err
}

func (c *FakeEvents) List(opts v1.ListOptions) (result *v1.EventList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewListAction(eventsResource, c.ns, opts), &v1.EventList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1.EventList{}
	for _, item := range obj.(*v1.EventList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested events.
func (c *FakeEvents) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewWatchAction(eventsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched event.
func (c *FakeEvents) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.Event, err error) {
	obj, err := c.Fake.
		Invokes(core.NewPatchSubresourceAction(eventsResource, c.ns, name, data, subresources...), &v1.Event{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Event), err
}
