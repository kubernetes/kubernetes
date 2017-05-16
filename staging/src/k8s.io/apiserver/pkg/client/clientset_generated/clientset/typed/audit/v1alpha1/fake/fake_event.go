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
	v1alpha1 "k8s.io/apiserver/pkg/apis/audit/v1alpha1"
	testing "k8s.io/client-go/testing"
)

// FakeEvents implements EventInterface
type FakeEvents struct {
	Fake *FakeAuditV1alpha1
}

var eventsResource = schema.GroupVersionResource{Group: "audit.k8s.io", Version: "v1alpha1", Resource: "events"}

var eventsKind = schema.GroupVersionKind{Group: "audit.k8s.io", Version: "v1alpha1", Kind: "Event"}

func (c *FakeEvents) Create(event *v1alpha1.Event) (result *v1alpha1.Event, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(eventsResource, event), &v1alpha1.Event{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.Event), err
}

func (c *FakeEvents) Update(event *v1alpha1.Event) (result *v1alpha1.Event, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(eventsResource, event), &v1alpha1.Event{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.Event), err
}

func (c *FakeEvents) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(eventsResource, name), &v1alpha1.Event{})
	return err
}

func (c *FakeEvents) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(eventsResource, listOptions)

	_, err := c.Fake.Invokes(action, &v1alpha1.EventList{})
	return err
}

func (c *FakeEvents) Get(name string, options v1.GetOptions) (result *v1alpha1.Event, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(eventsResource, name), &v1alpha1.Event{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.Event), err
}

func (c *FakeEvents) List(opts v1.ListOptions) (result *v1alpha1.EventList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(eventsResource, eventsKind, opts), &v1alpha1.EventList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1alpha1.EventList{}
	for _, item := range obj.(*v1alpha1.EventList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested events.
func (c *FakeEvents) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(eventsResource, opts))
}

// Patch applies the patch and returns the patched event.
func (c *FakeEvents) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1alpha1.Event, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(eventsResource, name, data, subresources...), &v1alpha1.Event{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.Event), err
}
