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

package fake

import (
	v1beta1 "k8s.io/api/events/v1beta1"
	types "k8s.io/apimachinery/pkg/types"
	core "k8s.io/client-go/testing"
)

// CreateWithEventNamespace creats a new event. Returns the copy of the event the server returns, or an error.
func (c *fakeEvents) CreateWithEventNamespace(event *v1beta1.Event) (*v1beta1.Event, error) {
	action := core.NewRootCreateAction(c.Resource(), event)
	if c.Namespace() != "" {
		action = core.NewCreateAction(c.Resource(), c.Namespace(), event)
	}
	obj, err := c.Fake.Invokes(action, event)
	if obj == nil {
		return nil, err
	}

	return obj.(*v1beta1.Event), err
}

// UpdateWithEventNamespace replaces an existing event. Returns the copy of the event the server returns, or an error.
func (c *fakeEvents) UpdateWithEventNamespace(event *v1beta1.Event) (*v1beta1.Event, error) {
	action := core.NewRootUpdateAction(c.Resource(), event)
	if c.Namespace() != "" {
		action = core.NewUpdateAction(c.Resource(), c.Namespace(), event)
	}
	obj, err := c.Fake.Invokes(action, event)
	if obj == nil {
		return nil, err
	}

	return obj.(*v1beta1.Event), err
}

// PatchWithEventNamespace patches an existing event. Returns the copy of the event the server returns, or an error.
func (c *fakeEvents) PatchWithEventNamespace(event *v1beta1.Event, data []byte) (*v1beta1.Event, error) {
	pt := types.StrategicMergePatchType
	action := core.NewRootPatchAction(c.Resource(), event.Name, pt, data)
	if c.Namespace() != "" {
		action = core.NewPatchAction(c.Resource(), c.Namespace(), event.Name, pt, data)
	}
	obj, err := c.Fake.Invokes(action, event)
	if obj == nil {
		return nil, err
	}

	return obj.(*v1beta1.Event), err
}
