/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package unversioned

import (
	api "k8s.io/kubernetes/pkg/api"
	unversioned "k8s.io/kubernetes/pkg/api/unversioned"
	watch "k8s.io/kubernetes/pkg/watch"
)

// EventNamespacer has methods to work with Event resources in a namespace
type EventNamespacer interface {
	Events(namespace string) EventInterface
}

// EventInterface has methods to work with Event resources.
type EventInterface interface {
	Create(*api.Event) (*api.Event, error)
	Update(*api.Event) (*api.Event, error)
	Delete(name string, options *api.DeleteOptions) error
	Get(name string) (*api.Event, error)
	List(opts unversioned.ListOptions) (*api.EventList, error)
	Watch(opts unversioned.ListOptions) (watch.Interface, error)
}

// events implements EventInterface
type events struct {
	client *LegacyClient
	ns     string
}

// newEvents returns a Events
func newEvents(c *LegacyClient, namespace string) *events {
	return &events{
		client: c,
		ns:     namespace,
	}
}

// Create takes the representation of a event and creates it.  Returns the server's representation of the event, and an error, if there is any.
func (c *events) Create(event *api.Event) (result *api.Event, err error) {
	result = &api.Event{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("events").
		Body(event).
		Do().
		Into(result)
	return
}

// Update takes the representation of a event and updates it. Returns the server's representation of the event, and an error, if there is any.
func (c *events) Update(event *api.Event) (result *api.Event, err error) {
	result = &api.Event{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("events").
		Name(event.Name).
		Body(event).
		Do().
		Into(result)
	return
}

// Delete takes name of the event and deletes it. Returns an error if one occurs.
func (c *events) Delete(name string, options *api.DeleteOptions) error {
	if options == nil {
		return c.client.Delete().Namespace(c.ns).Resource("events").Name(name).Do().Error()
	}
	body, err := api.Scheme.EncodeToVersion(options, c.client.APIVersion().String())
	if err != nil {
		return err
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("events").
		Name(name).
		Body(body).
		Do().
		Error()
}

// Get takes name of the event, and returns the corresponding event object, and an error if there is any.
func (c *events) Get(name string) (result *api.Event, err error) {
	result = &api.Event{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("events").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Events that match those selectors.
func (c *events) List(opts unversioned.ListOptions) (result *api.EventList, err error) {
	result = &api.EventList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("events").
		VersionedParams(&opts, api.Scheme).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested events.
func (c *events) Watch(opts unversioned.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("events").
		VersionedParams(&opts, api.Scheme).
		Watch()
}
