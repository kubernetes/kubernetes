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
	v1beta1 "k8s.io/api/events/v1beta1"
	types "k8s.io/apimachinery/pkg/types"
	core "k8s.io/client-go/testing"
)

// CreateWithEventNamespace creats a new event. Returns the copy of the event the server returns, or an error.
func (c *FakeEvents) CreateWithEventNamespace(event *v1beta1.Event) (*v1beta1.Event, error) {
	action := core.NewRootCreateAction(eventsResource, event)
	if c.ns != "" {
		action = core.NewCreateAction(eventsResource, c.ns, event)
	}
	obj, err := c.Fake.Invokes(action, event)
	if obj == nil {
		return nil, err
	}

	return obj.(*v1beta1.Event), err
}

// UpdateWithEventNamespace replaces an existing event. Returns the copy of the event the server returns, or an error.
func (c *FakeEvents) UpdateWithEventNamespace(event *v1beta1.Event) (*v1beta1.Event, error) {
	action := core.NewRootUpdateAction(eventsResource, event)
	if c.ns != "" {
		action = core.NewUpdateAction(eventsResource, c.ns, event)
	}
	obj, err := c.Fake.Invokes(action, event)
	if obj == nil {
		return nil, err
	}

	return obj.(*v1beta1.Event), err
}

// PatchWithEventNamespace patches an existing event. Returns the copy of the event the server returns, or an error.
func (c *FakeEvents) PatchWithEventNamespace(event *v1beta1.Event, data []byte) (*v1beta1.Event, error) {
	pt := types.StrategicMergePatchType
	action := core.NewRootPatchAction(eventsResource, event.Name, pt, data)
	if c.ns != "" {
		action = core.NewPatchAction(eventsResource, c.ns, event.Name, pt, data)
	}
	obj, err := c.Fake.Invokes(action, event)
	if obj == nil {
		return nil, err
	}

	return obj.(*v1beta1.Event), err
}
