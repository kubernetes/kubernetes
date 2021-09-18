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

package fake

import (
	"fmt"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	types "k8s.io/apimachinery/pkg/types"
	core "k8s.io/client-go/testing"
	ref "k8s.io/client-go/tools/reference"
)

// CreateWithEventNamespace makes a new event. Returns the copy of the event the server returns,
// or an error. The namespace to create the event within is deduced from the
// event; it must either match this event client's namespace, or this event
// client must have been created with the "" namespace.
func (c *FakeEvents) CreateWithEventNamespace(event *v1.Event) (*v1.Event, error) {
	if c.ns != "" && event.Namespace != c.ns {
		return nil, fmt.Errorf("can't create an event with namespace '%v' in namespace '%v'", event.Namespace, c.ns)
	}
	action := core.NewCreateAction(eventsResource, event.Namespace, event)
	obj, err := c.Fake.Invokes(action, event)
	if obj == nil {
		return nil, err
	}

	return obj.(*v1.Event), err
}

// UpdateWithEventNamespace modifies an existing event. It returns the copy of the event that the server returns,
// or an error. The namespace and key to update the event within is deduced from the event. The
// namespace must either match this event client's namespace, or this event client must have been
// created with the "" namespace.
func (c *FakeEvents) UpdateWithEventNamespace(event *v1.Event) (*v1.Event, error) {
	action := core.NewUpdateAction(eventsResource, event.Namespace, event)
	obj, err := c.Fake.Invokes(action, event)
	if obj == nil {
		return nil, err
	}

	return obj.(*v1.Event), err
}

// PatchWithEventNamespace modifies an existing event. It returns the copy of
// the event that the server returns, or an error. The namespace and name of the
// target event is deduced from the incompleteEvent. The namespace must either
// match this event client's namespace, or this event client must have been
// created with the "" namespace.
// TODO: Should take a PatchType as an argument probably.
func (c *FakeEvents) PatchWithEventNamespace(event *v1.Event, data []byte) (*v1.Event, error) {
	if c.ns != "" && event.Namespace != c.ns {
		return nil, fmt.Errorf("can't patch an event with namespace '%v' in namespace '%v'", event.Namespace, c.ns)
	}
	// TODO: Should be configurable to support additional patch strategies.
	pt := types.StrategicMergePatchType
	action := core.NewPatchAction(eventsResource, event.Namespace, event.Name, pt, data)
	obj, err := c.Fake.Invokes(action, event)
	if obj == nil {
		return nil, err
	}

	return obj.(*v1.Event), err
}

// Search returns a list of events matching the specified object.
func (c *FakeEvents) Search(scheme *runtime.Scheme, objOrRef runtime.Object) (*v1.EventList, error) {
	ref, err := ref.GetReference(scheme, objOrRef)
	if err != nil {
		return nil, err
	}
	if len(c.ns) > 0 && ref.Namespace != c.ns {
		return nil, fmt.Errorf("won't be able to find any events of namespace '%v' in namespace '%v'", ref.Namespace, c.ns)
	}
	action := core.NewListAction(eventsResource, eventsKind, ref.Namespace, metav1.ListOptions{})
	obj, err := c.Fake.Invokes(action, &v1.EventList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*v1.EventList), err
}

func (c *FakeEvents) GetFieldSelector(involvedObjectName, involvedObjectNamespace, involvedObjectKind, involvedObjectUID *string) fields.Selector {
	action := core.GenericActionImpl{}
	action.Verb = "get-field-selector"
	action.Resource = eventsResource

	c.Fake.Invokes(action, nil)
	return fields.Everything()
}
