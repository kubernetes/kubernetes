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
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	types "k8s.io/apimachinery/pkg/types"
	core "k8s.io/client-go/testing"
)

func (c *fakeEvents) CreateWithEventNamespace(event *v1.Event) (*v1.Event, error) {
	var action core.CreateActionImpl
	if c.Namespace() != "" {
		action = core.NewCreateAction(c.Resource(), c.Namespace(), event)
	} else {
		action = core.NewCreateAction(c.Resource(), event.GetNamespace(), event)
	}
	obj, err := c.Fake.Invokes(action, event)
	if obj == nil {
		return nil, err
	}

	return obj.(*v1.Event), err
}

// Update replaces an existing event. Returns the copy of the event the server returns, or an error.
func (c *fakeEvents) UpdateWithEventNamespace(event *v1.Event) (*v1.Event, error) {
	var action core.UpdateActionImpl
	if c.Namespace() != "" {
		action = core.NewUpdateAction(c.Resource(), c.Namespace(), event)
	} else {
		action = core.NewUpdateAction(c.Resource(), event.GetNamespace(), event)
	}
	obj, err := c.Fake.Invokes(action, event)
	if obj == nil {
		return nil, err
	}

	return obj.(*v1.Event), err
}

// PatchWithEventNamespace patches an existing event. Returns the copy of the event the server returns, or an error.
// TODO: Should take a PatchType as an argument probably.
func (c *fakeEvents) PatchWithEventNamespace(event *v1.Event, data []byte) (*v1.Event, error) {
	// TODO: Should be configurable to support additional patch strategies.
	pt := types.StrategicMergePatchType
	var action core.PatchActionImpl
	if c.Namespace() != "" {
		action = core.NewPatchAction(c.Resource(), c.Namespace(), event.Name, pt, data)
	} else {
		action = core.NewPatchAction(c.Resource(), event.GetNamespace(), event.Name, pt, data)
	}
	obj, err := c.Fake.Invokes(action, event)
	if obj == nil {
		return nil, err
	}

	return obj.(*v1.Event), err
}

// Search returns a list of events matching the specified object.
func (c *fakeEvents) Search(scheme *runtime.Scheme, objOrRef runtime.Object) (*v1.EventList, error) {
	var action core.ListActionImpl
	if c.Namespace() != "" {
		action = core.NewListAction(c.Resource(), c.Kind(), c.Namespace(), metav1.ListOptions{})
	} else {
		action = core.NewListAction(c.Resource(), c.Kind(), v1.NamespaceDefault, metav1.ListOptions{})
	}
	obj, err := c.Fake.Invokes(action, &v1.EventList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*v1.EventList), err
}

func (c *fakeEvents) GetFieldSelector(involvedObjectName, involvedObjectNamespace, involvedObjectKind, involvedObjectUID *string) fields.Selector {
	action := core.GenericActionImpl{}
	action.Verb = "get-field-selector"
	action.Resource = c.Resource()

	c.Fake.Invokes(action, nil)
	return fields.Everything()
}
