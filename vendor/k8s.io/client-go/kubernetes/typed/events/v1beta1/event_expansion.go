/*
Copyright 2019 The Kubernetes Authors.

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

package v1beta1

import (
	"context"
	"fmt"

	"k8s.io/api/events/v1beta1"
	"k8s.io/apimachinery/pkg/types"
)

// The EventExpansion interface allows manually adding extra methods to the EventInterface.
// TODO: Add querying functions to the event expansion
type EventExpansion interface {
	// CreateWithEventNamespace is the same as a Create
	// except that it sends the request to the event.Namespace.
	CreateWithEventNamespace(event *v1beta1.Event) (*v1beta1.Event, error)
	// UpdateWithEventNamespace is the same as a Update
	// except that it sends the request to the event.Namespace.
	UpdateWithEventNamespace(event *v1beta1.Event) (*v1beta1.Event, error)
	// PatchWithEventNamespace is the same as a Patch
	// except that it sends the request to the event.Namespace.
	PatchWithEventNamespace(event *v1beta1.Event, data []byte) (*v1beta1.Event, error)
}

// CreateWithEventNamespace makes a new event.
// Returns the copy of the event the server returns, or an error.
// The namespace to create the event within is deduced from the event.
// it must either match this event client's namespace, or this event client must
// have been created with the "" namespace.
func (e *events) CreateWithEventNamespace(event *v1beta1.Event) (*v1beta1.Event, error) {
	if e.ns != "" && event.Namespace != e.ns {
		return nil, fmt.Errorf("can't create an event with namespace '%v' in namespace '%v'", event.Namespace, e.ns)
	}
	result := &v1beta1.Event{}
	err := e.client.Post().
		NamespaceIfScoped(event.Namespace, len(event.Namespace) > 0).
		Resource("events").
		Body(event).
		Do(context.TODO()).
		Into(result)
	return result, err
}

// UpdateWithEventNamespace modifies an existing event.
// It returns the copy of the event that the server returns, or an error.
// The namespace and key to update the event within is deduced from the event.
// The namespace must either match this event client's namespace, or this event client must have been
// created with the "" namespace.
// Update also requires the ResourceVersion to be set in the event object.
func (e *events) UpdateWithEventNamespace(event *v1beta1.Event) (*v1beta1.Event, error) {
	if e.ns != "" && event.Namespace != e.ns {
		return nil, fmt.Errorf("can't update an event with namespace '%v' in namespace '%v'", event.Namespace, e.ns)
	}
	result := &v1beta1.Event{}
	err := e.client.Put().
		NamespaceIfScoped(event.Namespace, len(event.Namespace) > 0).
		Resource("events").
		Name(event.Name).
		Body(event).
		Do(context.TODO()).
		Into(result)
	return result, err
}

// PatchWithEventNamespace modifies an existing event.
// It returns the copy of the event that the server returns, or an error.
// The namespace and name of the target event is deduced from the event.
// The namespace must either match this event client's namespace, or this event client must
//  have been created with the "" namespace.
func (e *events) PatchWithEventNamespace(event *v1beta1.Event, data []byte) (*v1beta1.Event, error) {
	if e.ns != "" && event.Namespace != e.ns {
		return nil, fmt.Errorf("can't patch an event with namespace '%v' in namespace '%v'", event.Namespace, e.ns)
	}
	result := &v1beta1.Event{}
	err := e.client.Patch(types.StrategicMergePatchType).
		NamespaceIfScoped(event.Namespace, len(event.Namespace) > 0).
		Resource("events").
		Name(event.Name).
		Body(data).
		Do(context.TODO()).
		Into(result)
	return result, err
}
