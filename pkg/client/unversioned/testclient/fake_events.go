/*
Copyright 2014 The Kubernetes Authors.

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

package testclient

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

// FakeEvents implements EventInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeEvents struct {
	Fake      *Fake
	Namespace string
}

// Get returns the given event, or an error.
func (c *FakeEvents) Get(name string) (*api.Event, error) {
	action := NewRootGetAction("events", name)
	if c.Namespace != "" {
		action = NewGetAction("events", c.Namespace, name)
	}
	obj, err := c.Fake.Invokes(action, &api.Event{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Event), err
}

// List returns a list of events matching the selectors.
func (c *FakeEvents) List(opts api.ListOptions) (*api.EventList, error) {
	action := NewRootListAction("events", opts)
	if c.Namespace != "" {
		action = NewListAction("events", c.Namespace, opts)
	}
	obj, err := c.Fake.Invokes(action, &api.EventList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.EventList), err
}

// Create makes a new event. Returns the copy of the event the server returns, or an error.
func (c *FakeEvents) Create(event *api.Event) (*api.Event, error) {
	action := NewRootCreateAction("events", event)
	if c.Namespace != "" {
		action = NewCreateAction("events", c.Namespace, event)
	}
	obj, err := c.Fake.Invokes(action, event)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Event), err
}

// Update replaces an existing event. Returns the copy of the event the server returns, or an error.
func (c *FakeEvents) Update(event *api.Event) (*api.Event, error) {
	action := NewRootUpdateAction("events", event)
	if c.Namespace != "" {
		action = NewUpdateAction("events", c.Namespace, event)
	}
	obj, err := c.Fake.Invokes(action, event)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Event), err
}

// Patch patches an existing event. Returns the copy of the event the server returns, or an error.
func (c *FakeEvents) Patch(event *api.Event, data []byte) (*api.Event, error) {
	action := NewRootPatchAction("events", event)
	if c.Namespace != "" {
		action = NewPatchAction("events", c.Namespace, event)
	}
	obj, err := c.Fake.Invokes(action, event)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Event), err
}

func (c *FakeEvents) Delete(name string) error {
	action := NewRootDeleteAction("events", name)
	if c.Namespace != "" {
		action = NewDeleteAction("events", c.Namespace, name)
	}
	_, err := c.Fake.Invokes(action, &api.Event{})
	return err
}

func (c *FakeEvents) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := NewRootDeleteCollectionAction("events", listOptions)
	if c.Namespace != "" {
		action = NewDeleteCollectionAction("events", c.Namespace, listOptions)
	}
	_, err := c.Fake.Invokes(action, &api.EventList{})
	return err
}

// Watch starts watching for events matching the given selectors.
func (c *FakeEvents) Watch(opts api.ListOptions) (watch.Interface, error) {
	action := NewRootWatchAction("events", opts)
	if c.Namespace != "" {
		action = NewWatchAction("events", c.Namespace, opts)
	}
	return c.Fake.InvokesWatch(action)
}

// Search returns a list of events matching the specified object.
func (c *FakeEvents) Search(objOrRef runtime.Object) (*api.EventList, error) {
	action := NewRootListAction("events", api.ListOptions{})
	if c.Namespace != "" {
		action = NewListAction("events", c.Namespace, api.ListOptions{})
	}
	obj, err := c.Fake.Invokes(action, &api.EventList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.EventList), err
}

func (c *FakeEvents) GetFieldSelector(involvedObjectName, involvedObjectNamespace, involvedObjectKind, involvedObjectUID *string) fields.Selector {
	action := GenericActionImpl{}
	action.Verb = "get-field-selector"
	action.Resource = "events"

	c.Fake.Invokes(action, nil)
	return fields.Everything()
}
