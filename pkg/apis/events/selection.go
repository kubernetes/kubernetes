/*
Copyright 2024 The Kubernetes Authors.

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

package events

import (
	"fmt"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/core"
)

func init() {
	SchemeBuilder.Register(addSelectorFuncs)
}

// addSelectorFuncs adds versioned selector funcs for resources to the scheme.
func addSelectorFuncs(scheme *runtime.Scheme) error {
	scheme.AddSelectorFunc(SchemeGroupVersion.WithKind("Event"), EventSelectorFunc)
	return nil
}

// EventSelectorFunc returns true if the object is an event that matches the label and field selectors.
func EventSelectorFunc(obj runtime.Object, selector runtime.Selectors) (bool, error) {
	return EventMatcher(selector).Matches(obj)
}

// EventMatcher returns a selection predicate for a given label and field selector.
func EventMatcher(selector runtime.Selectors) runtime.SelectionPredicate {
	return runtime.SelectionPredicate{
		Selectors: selector,
		GetAttrs:  EventGetAttrs,
	}
}

// EventGetAttrs returns labels and fields of a given object for filtering purposes.
func EventGetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	event, ok := obj.(*core.Event)
	if !ok {
		return nil, nil, fmt.Errorf("not an event")
	}
	return labels.Set(event.Labels), EventToSelectableFields(event), nil
}

// EventToSelectableFields returns a field set that represents the object.
func EventToSelectableFields(event *core.Event) fields.Set {
	defaultFields := runtime.DefaultSelectableFields(event)
	// Use exact size to reduce memory allocations.
	// If you change the fields, adjust the size.
	specificFieldsSet := make(fields.Set, len(defaultFields)+11)
	source := event.Source.Component
	if source == "" {
		source = event.ReportingController
	}
	specificFieldsSet["involvedObject.kind"] = event.InvolvedObject.Kind
	specificFieldsSet["involvedObject.namespace"] = event.InvolvedObject.Namespace
	specificFieldsSet["involvedObject.name"] = event.InvolvedObject.Name
	specificFieldsSet["involvedObject.uid"] = string(event.InvolvedObject.UID)
	specificFieldsSet["involvedObject.apiVersion"] = event.InvolvedObject.APIVersion
	specificFieldsSet["involvedObject.resourceVersion"] = event.InvolvedObject.ResourceVersion
	specificFieldsSet["involvedObject.fieldPath"] = event.InvolvedObject.FieldPath
	specificFieldsSet["reason"] = event.Reason
	specificFieldsSet["reportingComponent"] = event.ReportingController // use the core/v1 field name
	specificFieldsSet["source"] = source
	specificFieldsSet["type"] = event.Type
	return runtime.MergeFieldsSets(specificFieldsSet, defaultFields)
}
