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

// Events has methods to work with Event resources
type EventsInterface interface {
	Events() EventInterface
}

// EventInterface has methods to work with Event resources
type EventInterface interface {
	Create(event *api.Event) (*api.Event, error)
	List(label, field labels.Selector) (*api.EventList, error)
	Get(id string) (*api.Event, error)
	Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error)
}

// events implements Events interface
type events struct {
	r *Client
}

// newEvents returns a events
func newEvents(c *Client) *events {
	return &events{
		r: c,
	}
}

// Create makes a new event. Returns the copy of the event the server returns, or an error.
func (c *events) Create(event *api.Event) (*api.Event, error) {
	result := &api.Event{}
	err := c.r.Post().Path("events").Namespace(event.Namespace).Body(event).Do().Into(result)
	return result, err
}

// List returns a list of events matching the selectors.
func (c *events) List(label, field labels.Selector) (*api.EventList, error) {
	result := &api.EventList{}
	err := c.r.Get().
		Path("events").
		SelectorParam("labels", label).
		SelectorParam("fields", field).
		Do().
		Into(result)
	return result, err
}

// Get returns the given event, or an error.
func (c *events) Get(id string) (*api.Event, error) {
	result := &api.Event{}
	err := c.r.Get().Path("events").Path(id).Do().Into(result)
	return result, err
}

// Watch starts watching for events matching the given selectors.
func (c *events) Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	return c.r.Get().
		Path("watch").
		Path("events").
		Param("resourceVersion", resourceVersion).
		SelectorParam("labels", label).
		SelectorParam("fields", field).
		Watch()
}
