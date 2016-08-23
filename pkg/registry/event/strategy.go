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

package event

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

type eventStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that pplies when creating and updating
// Event objects via the REST API.
var Strategy = eventStrategy{api.Scheme, api.SimpleNameGenerator}

func (eventStrategy) NamespaceScoped() bool {
	return true
}

func (eventStrategy) PrepareForCreate(ctx api.Context, obj runtime.Object) {
}

func (eventStrategy) PrepareForUpdate(ctx api.Context, obj, old runtime.Object) {
}

func (eventStrategy) Validate(ctx api.Context, obj runtime.Object) field.ErrorList {
	event := obj.(*api.Event)
	return validation.ValidateEvent(event)
}

// Canonicalize normalizes the object after validation.
func (eventStrategy) Canonicalize(obj runtime.Object) {
}

func (eventStrategy) AllowCreateOnUpdate() bool {
	return true
}

func (eventStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	event := obj.(*api.Event)
	return validation.ValidateEvent(event)
}

func (eventStrategy) AllowUnconditionalUpdate() bool {
	return true
}

func MatchEvent(label labels.Selector, field fields.Selector) *generic.SelectionPredicate {
	return &generic.SelectionPredicate{
		Label: label,
		Field: field,
		GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
			event, ok := obj.(*api.Event)
			if !ok {
				return nil, nil, fmt.Errorf("not an event")
			}
			return labels.Set(event.Labels), SelectableFields(event), nil
		},
	}
}

func SelectableFields(event *api.Event) fields.Set {
	fields := make(fields.Set, 12)
	fields["involvedObject.kind"] = event.InvolvedObject.Kind
	fields["involvedObject.namespace"] = event.InvolvedObject.Namespace
	fields["involvedObject.name"] = event.InvolvedObject.Name
	fields["involvedObject.uid"] = string(event.InvolvedObject.UID)
	fields["involvedObject.apiVersion"] = event.InvolvedObject.APIVersion
	fields["involvedObject.resourceVersion"] = event.InvolvedObject.ResourceVersion
	fields["involvedObject.fieldPath"] = event.InvolvedObject.FieldPath
	fields["reason"] = event.Reason
	fields["source"] = event.Source.Component
	fields["type"] = event.Type
	generic.AddObjectMetaFields(&event.ObjectMeta, true, &fields)
	return fields
}
