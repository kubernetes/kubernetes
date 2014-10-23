/*
Copyright 2014 Google Inc. All rights reserved.

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

package client

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// FakeEvents implements EventInterface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeEvents struct {
	Fake *Fake
}

// Create makes a new event. Returns the copy of the event the server returns, or an error.
func (c *FakeEvents) Create(event *api.Event) (*api.Event, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "get-event", Value: event.Name})
	return &api.Event{}, nil
}

// List returns a list of events matching the selectors.
func (c *FakeEvents) List(label, field labels.Selector) (*api.EventList, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "list-events"})
	return &c.Fake.EventsList, nil
}

// Get returns the given event, or an error.
func (c *FakeEvents) Get(id string) (*api.Event, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "get-event", Value: id})
	return &api.Event{}, nil
}

// Watch starts watching for events matching the given selectors.
func (c *FakeEvents) Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	c.Fake.Actions = append(c.Fake.Actions, FakeAction{Action: "watch-events", Value: resourceVersion})
	return c.Fake.Watch, c.Fake.Err
}
