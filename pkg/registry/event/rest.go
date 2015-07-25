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

package event

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// REST adapts an event registry into apiserver's RESTStorage model.
type REST struct {
	registry generic.Registry
}

// NewStorage returns a new REST. You must use a registry created by
// NewEtcdRegistry unless you're testing.
func NewStorage(registry generic.Registry) *REST {
	return &REST{
		registry: registry,
	}
}

func (rs *REST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	event, ok := obj.(*api.Event)
	if !ok {
		return nil, errors.NewInternalError(fmt.Errorf("received object is not of type event: %#v", obj))
	}
	if api.NamespaceValue(ctx) != "" {
		if !api.ValidNamespace(ctx, &event.ObjectMeta) {
			return nil, errors.NewConflict("event", event.Namespace, fmt.Errorf("event.namespace does not match the provided context"))
		}
	}
	if errs := validation.ValidateEvent(event); len(errs) > 0 {
		return nil, errors.NewInvalid("event", event.Name, errs)
	}
	api.FillObjectMetaSystemFields(ctx, &event.ObjectMeta)

	err := rs.registry.CreateWithName(ctx, event.Name, event)
	if err != nil {
		return nil, err
	}
	return rs.registry.Get(ctx, event.Name)
}

// Update replaces an existing Event instance in storage.registry, with the given instance.
func (rs *REST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	event, ok := obj.(*api.Event)
	if !ok {
		return nil, false, errors.NewInternalError(fmt.Errorf("received object is not of type event: %#v", obj))

	}
	if api.NamespaceValue(ctx) != "" {
		if !api.ValidNamespace(ctx, &event.ObjectMeta) {
			return nil, false, errors.NewConflict("event", event.Namespace, fmt.Errorf("event.namespace does not match the provided context"))
		}
	}
	if errs := validation.ValidateEvent(event); len(errs) > 0 {
		return nil, false, errors.NewInvalid("event", event.Name, errs)
	}
	api.FillObjectMetaSystemFields(ctx, &event.ObjectMeta)

	err := rs.registry.UpdateWithName(ctx, event.Name, event)
	if err != nil {
		return nil, false, err
	}
	out, err := rs.registry.Get(ctx, event.Name)
	return out, false, err
}

func (rs *REST) Delete(ctx api.Context, name string) (runtime.Object, error) {
	obj, err := rs.registry.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	_, ok := obj.(*api.Event)
	if !ok {
		return nil, errors.NewInternalError(fmt.Errorf("stored object %s is not of type event: %#v", name, obj))
	}
	return rs.registry.Delete(ctx, name, nil)
}

func (rs *REST) Get(ctx api.Context, name string) (runtime.Object, error) {
	obj, err := rs.registry.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	event, ok := obj.(*api.Event)
	if !ok {
		return nil, errors.NewInternalError(fmt.Errorf("stored object %s is not of type event: %#v", name, obj))
	}
	return event, err
}

func (rs *REST) getAttrs(obj runtime.Object) (objLabels labels.Set, objFields fields.Set, err error) {
	event, ok := obj.(*api.Event)
	if !ok {
		return nil, nil, errors.NewInternalError(fmt.Errorf("object is not of type event: %#v", obj))
	}
	l := event.Labels
	if l == nil {
		l = labels.Set{}
	}
	return l, fields.Set{
		"metadata.name":                  event.Name,
		"involvedObject.kind":            event.InvolvedObject.Kind,
		"involvedObject.namespace":       event.InvolvedObject.Namespace,
		"involvedObject.name":            event.InvolvedObject.Name,
		"involvedObject.uid":             string(event.InvolvedObject.UID),
		"involvedObject.apiVersion":      event.InvolvedObject.APIVersion,
		"involvedObject.resourceVersion": fmt.Sprintf("%s", event.InvolvedObject.ResourceVersion),
		"involvedObject.fieldPath":       event.InvolvedObject.FieldPath,
		"reason":                         event.Reason,
		"source":                         event.Source.Component,
	}, nil
}

func (rs *REST) List(ctx api.Context, label labels.Selector, field fields.Selector) (runtime.Object, error) {
	return rs.registry.ListPredicate(ctx, &generic.SelectionPredicate{label, field, rs.getAttrs})
}

// Watch returns Events events via a watch.Interface.
// It implements rest.Watcher.
func (rs *REST) Watch(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return rs.registry.WatchPredicate(ctx, &generic.SelectionPredicate{label, field, rs.getAttrs}, resourceVersion)
}

// New returns a new api.Event
func (*REST) New() runtime.Object {
	return &api.Event{}
}

func (*REST) NewList() runtime.Object {
	return &api.EventList{}
}
