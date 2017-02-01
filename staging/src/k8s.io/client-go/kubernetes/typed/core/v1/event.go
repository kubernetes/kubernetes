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

package v1

import (
	api "k8s.io/client-go/pkg/api"
	v1 "k8s.io/client-go/pkg/api/v1"
	watch "k8s.io/client-go/pkg/watch"
	rest "k8s.io/client-go/rest"
)

// EventsGetter has a method to return a EventInterface.
// A group's client should implement this interface.
type EventsGetter interface {
	Events(namespace string) EventInterface
}

// EventInterface has methods to work with Event resources.
type EventInterface interface {
	Create(*v1.Event) (*v1.Event, error)
	Update(*v1.Event) (*v1.Event, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string) (*v1.Event, error)
	List(opts v1.ListOptions) (*v1.EventList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.Event, err error)
	EventExpansion
}

// events implements EventInterface
type events struct {
	client rest.Interface
	ns     string
}

// newEvents returns a Events
func newEvents(c *CoreV1Client, namespace string) *events {
	return &events{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Create takes the representation of a event and creates it.  Returns the server's representation of the event, and an error, if there is any.
func (c *events) Create(event *v1.Event) (result *v1.Event, err error) {
	result = &v1.Event{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("events").
		Body(event).
		Do().
		Into(result)
	return
}

// Update takes the representation of a event and updates it. Returns the server's representation of the event, and an error, if there is any.
func (c *events) Update(event *v1.Event) (result *v1.Event, err error) {
	result = &v1.Event{}
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
func (c *events) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("events").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *events) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("events").
		VersionedParams(&listOptions, api.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Get takes name of the event, and returns the corresponding event object, and an error if there is any.
func (c *events) Get(name string) (result *v1.Event, err error) {
	result = &v1.Event{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("events").
		Name(name).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of Events that match those selectors.
func (c *events) List(opts v1.ListOptions) (result *v1.EventList, err error) {
	result = &v1.EventList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("events").
		VersionedParams(&opts, api.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested events.
func (c *events) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(c.ns).
		Resource("events").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}

// Patch applies the patch and returns the patched event.
func (c *events) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1.Event, err error) {
	result = &v1.Event{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("events").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
