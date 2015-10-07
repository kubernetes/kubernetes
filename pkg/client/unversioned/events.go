/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

// EventNamespacer can return an EventInterface for the given namespace.
type EventNamespacer interface {
	Events(namespace string) EventInterface
}

// EventInterface has methods to work with Event resources
type EventInterface interface {
	Create(event *api.Event) (*api.Event, error)
	Update(event *api.Event) (*api.Event, error)
	Patch(event *api.Event, data []byte) (*api.Event, error)
	List(label labels.Selector, field fields.Selector) (*api.EventList, error)
	Get(name string) (*api.Event, error)
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
	// Search finds events about the specified object
	Search(objOrRef runtime.Object) (*api.EventList, error)
	Delete(name string) error
	// Returns the appropriate field selector based on the API version being used to communicate with the server.
	// The returned field selector can be used with List and Watch to filter desired events.
	GetFieldSelector(involvedObjectName, involvedObjectNamespace, involvedObjectKind, involvedObjectUID *string) fields.Selector
}

// events implements Events interface
type events struct {
	client    *Client
	namespace string
}

// newEvents returns a new events object.
func newEvents(c *Client, ns string) *events {
	return &events{
		client:    c,
		namespace: ns,
	}
}

// Create makes a new event. Returns the copy of the event the server returns,
// or an error. The namespace to create the event within is deduced from the
// event; it must either match this event client's namespace, or this event
// client must have been created with the "" namespace.
func (e *events) Create(event *api.Event) (*api.Event, error) {
	if e.namespace != "" && event.Namespace != e.namespace {
		return nil, fmt.Errorf("can't create an event with namespace '%v' in namespace '%v'", event.Namespace, e.namespace)
	}
	result := &api.Event{}
	err := e.client.Post().
		NamespaceIfScoped(event.Namespace, len(event.Namespace) > 0).
		Resource("events").
		Body(event).
		Do().
		Into(result)
	return result, err
}

// Update modifies an existing event. It returns the copy of the event that the server returns,
// or an error. The namespace and key to update the event within is deduced from the event. The
// namespace must either match this event client's namespace, or this event client must have been
// created with the "" namespace. Update also requires the ResourceVersion to be set in the event
// object.
func (e *events) Update(event *api.Event) (*api.Event, error) {
	if len(event.ResourceVersion) == 0 {
		return nil, fmt.Errorf("invalid event update object, missing resource version: %#v", event)
	}
	result := &api.Event{}
	err := e.client.Put().
		NamespaceIfScoped(event.Namespace, len(event.Namespace) > 0).
		Resource("events").
		Name(event.Name).
		Body(event).
		Do().
		Into(result)
	return result, err
}

// Patch modifies an existing event. It returns the copy of the event that the server returns, or an
// error. The namespace and name of the target event is deduced from the incompleteEvent. The
// namespace must either match this event client's namespace, or this event client must have been
// created with the "" namespace.
func (e *events) Patch(incompleteEvent *api.Event, data []byte) (*api.Event, error) {
	result := &api.Event{}
	err := e.client.Patch(api.StrategicMergePatchType).
		NamespaceIfScoped(incompleteEvent.Namespace, len(incompleteEvent.Namespace) > 0).
		Resource("events").
		Name(incompleteEvent.Name).
		Body(data).
		Do().
		Into(result)
	return result, err
}

// List returns a list of events matching the selectors.
func (e *events) List(label labels.Selector, field fields.Selector) (*api.EventList, error) {
	result := &api.EventList{}
	err := e.client.Get().
		NamespaceIfScoped(e.namespace, len(e.namespace) > 0).
		Resource("events").
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Do().
		Into(result)
	return result, err
}

// Get returns the given event, or an error.
func (e *events) Get(name string) (*api.Event, error) {
	result := &api.Event{}
	err := e.client.Get().
		NamespaceIfScoped(e.namespace, len(e.namespace) > 0).
		Resource("events").
		Name(name).
		Do().
		Into(result)
	return result, err
}

// Watch starts watching for events matching the given selectors.
func (e *events) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return e.client.Get().
		Prefix("watch").
		NamespaceIfScoped(e.namespace, len(e.namespace) > 0).
		Resource("events").
		Param("resourceVersion", resourceVersion).
		LabelsSelectorParam(label).
		FieldsSelectorParam(field).
		Watch()
}

// Search finds events about the specified object. The namespace of the
// object must match this event's client namespace unless the event client
// was made with the "" namespace.
func (e *events) Search(objOrRef runtime.Object) (*api.EventList, error) {
	ref, err := api.GetReference(objOrRef)
	if err != nil {
		return nil, err
	}
	if e.namespace != "" && ref.Namespace != e.namespace {
		return nil, fmt.Errorf("won't be able to find any events of namespace '%v' in namespace '%v'", ref.Namespace, e.namespace)
	}
	stringRefKind := string(ref.Kind)
	var refKind *string
	if stringRefKind != "" {
		refKind = &stringRefKind
	}
	stringRefUID := string(ref.UID)
	var refUID *string
	if stringRefUID != "" {
		refUID = &stringRefUID
	}
	fieldSelector := e.GetFieldSelector(&ref.Name, &ref.Namespace, refKind, refUID)
	return e.List(labels.Everything(), fieldSelector)
}

// Delete deletes an existing event.
func (e *events) Delete(name string) error {
	return e.client.Delete().
		NamespaceIfScoped(e.namespace, len(e.namespace) > 0).
		Resource("events").
		Name(name).
		Do().
		Error()
}

// Returns the appropriate field selector based on the API version being used to communicate with the server.
// The returned field selector can be used with List and Watch to filter desired events.
func (e *events) GetFieldSelector(involvedObjectName, involvedObjectNamespace, involvedObjectKind, involvedObjectUID *string) fields.Selector {
	apiVersion := e.client.APIVersion()
	field := fields.Set{}
	if involvedObjectName != nil {
		field[getInvolvedObjectNameFieldLabel(apiVersion)] = *involvedObjectName
	}
	if involvedObjectNamespace != nil {
		field["involvedObject.namespace"] = *involvedObjectNamespace
	}
	if involvedObjectKind != nil {
		field["involvedObject.kind"] = *involvedObjectKind
	}
	if involvedObjectUID != nil {
		field["involvedObject.uid"] = *involvedObjectUID
	}
	return field.AsSelector()
}

// Returns the appropriate field label to use for name of the involved object as per the given API version.
func getInvolvedObjectNameFieldLabel(version string) string {
	return "involvedObject.name"
}
