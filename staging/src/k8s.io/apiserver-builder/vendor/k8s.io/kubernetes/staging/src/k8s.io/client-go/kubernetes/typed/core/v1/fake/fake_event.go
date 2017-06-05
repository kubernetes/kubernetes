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
	v1 "k8s.io/client-go/pkg/api/v1"
	testing "k8s.io/client-go/testing"
)

// FakeEvents implements EventInterface
type FakeEvents struct {
	Fake *FakeCoreV1
	ns   string
}

var eventsResource = schema.GroupVersionResource{Group: "", Version: "v1", Resource: "events"}

func (c *FakeEvents) Create(event *v1.Event) (result *v1.Event, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(eventsResource, c.ns, event), &v1.Event{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Event), err
}

func (c *FakeEvents) Update(event *v1.Event) (result *v1.Event, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(eventsResource, c.ns, event), &v1.Event{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Event), err
}

func (c *FakeEvents) Delete(name string, options *meta_v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(eventsResource, c.ns, name), &v1.Event{})

	return err
}

func (c *FakeEvents) DeleteCollection(options *meta_v1.DeleteOptions, listOptions meta_v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(eventsResource, c.ns, listOptions)

	_, err := c.Fake.Invokes(action, &v1.EventList{})
	return err
}

func (c *FakeEvents) Get(name string, options meta_v1.GetOptions) (result *v1.Event, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(eventsResource, c.ns, name), &v1.Event{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Event), err
}

func (c *FakeEvents) List(opts meta_v1.ListOptions) (result *v1.EventList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(eventsResource, c.ns, opts), &v1.EventList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
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
func (c *FakeEvents) Watch(opts meta_v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(eventsResource, c.ns, opts))

}

// Patch applies the patch and returns the patched event.
func (c *FakeEvents) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1.Event, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(eventsResource, c.ns, name, data, subresources...), &v1.Event{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1.Event), err
}
