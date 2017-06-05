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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/api"
)

func (c *FakeEvents) CreateWithEventNamespace(event *api.Event) (*api.Event, error) {
	action := core.NewRootCreateAction(eventsResource, event)
	if c.ns != "" {
		action = core.NewCreateAction(eventsResource, c.ns, event)
	}
	obj, err := c.Fake.Invokes(action, event)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Event), err
}

// Update replaces an existing event. Returns the copy of the event the server returns, or an error.
func (c *FakeEvents) UpdateWithEventNamespace(event *api.Event) (*api.Event, error) {
	action := core.NewRootUpdateAction(eventsResource, event)
	if c.ns != "" {
		action = core.NewUpdateAction(eventsResource, c.ns, event)
	}
	obj, err := c.Fake.Invokes(action, event)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Event), err
}

// PatchWithEventNamespace patches an existing event. Returns the copy of the event the server returns, or an error.
func (c *FakeEvents) PatchWithEventNamespace(event *api.Event, data []byte) (*api.Event, error) {
	action := core.NewRootPatchAction(eventsResource, event.Name, data)
	if c.ns != "" {
		action = core.NewPatchAction(eventsResource, c.ns, event.Name, data)
	}
	obj, err := c.Fake.Invokes(action, event)
	if obj == nil {
		return nil, err
	}

	return obj.(*api.Event), err
}

// Search returns a list of events matching the specified object.
func (c *FakeEvents) Search(scheme *runtime.Scheme, objOrRef runtime.Object) (*api.EventList, error) {
	action := core.NewRootListAction(eventsResource, metav1.ListOptions{})
	if c.ns != "" {
		action = core.NewListAction(eventsResource, c.ns, metav1.ListOptions{})
	}
	obj, err := c.Fake.Invokes(action, &api.EventList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*api.EventList), err
}

func (c *FakeEvents) GetFieldSelector(involvedObjectName, involvedObjectNamespace, involvedObjectKind, involvedObjectUID *string) fields.Selector {
	action := core.GenericActionImpl{}
	action.Verb = "get-field-selector"
	action.Resource = eventsResource

	c.Fake.Invokes(action, nil)
	return fields.Everything()
}
