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

package v1

import (
	"fmt"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
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
	event, ok := obj.(*Event)
	if !ok {
		return nil, nil, fmt.Errorf("not an event")
	}
	return labels.Set(event.Labels), EventToSelectableFields(event), nil
}

// EventToSelectableFields returns a field set that represents the object.
func EventToSelectableFields(event *Event) fields.Set {
	defaultFields := runtime.DefaultSelectableFields(event)
	// Use exact size to reduce memory allocations.
	// If you change the fields, adjust the size.
	specificFieldsSet := make(fields.Set, len(defaultFields)+11)
	source := event.DeprecatedSource.Component
	if source == "" {
		source = event.ReportingController
	}
	// "involvedObject" in core was renamed to "regarding" in events
	specificFieldsSet["regarding.kind"] = event.Regarding.Kind
	specificFieldsSet["regarding.namespace"] = event.Regarding.Namespace
	specificFieldsSet["regarding.name"] = event.Regarding.Name
	specificFieldsSet["regarding.uid"] = string(event.Regarding.UID)
	specificFieldsSet["regarding.apiVersion"] = event.Regarding.APIVersion
	specificFieldsSet["regarding.resourceVersion"] = event.Regarding.ResourceVersion
	specificFieldsSet["regarding.fieldPath"] = event.Regarding.FieldPath
	specificFieldsSet["reason"] = event.Reason
	specificFieldsSet["reportingController"] = event.ReportingController
	specificFieldsSet["source"] = source // TODO: Do we still need to support selection by source? There's no source field any more.
	specificFieldsSet["type"] = event.Type
	return runtime.MergeFieldsSets(specificFieldsSet, defaultFields)
}
