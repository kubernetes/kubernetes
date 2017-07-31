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

package validation

import (
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api"
	apiutil "k8s.io/kubernetes/pkg/api/util"
)

// ValidateEvent makes sure that the event makes sense.
func ValidateEvent(event *api.Event) field.ErrorList {
	allErrs := field.ErrorList{}

	zeroTime := time.Time{}

	// "New" Events require EventTime to be set, so if it's empty it means we're processing "old" Event.
	if event.EventTime.Time == zeroTime {
		// Make sure event.Namespace and the involvedObject.Namespace agree
		if len(event.InvolvedObject.Namespace) == 0 {
			// event.Namespace must also be empty (or "default", for compatibility with old clients)
			if event.Namespace != metav1.NamespaceNone && event.Namespace != metav1.NamespaceDefault {
				allErrs = append(allErrs, field.Invalid(field.NewPath("involvedObject", "namespace"), event.InvolvedObject.Namespace, "does not match event.namespace"))
			}
		} else {
			// event namespace must match
			if event.Namespace != event.InvolvedObject.Namespace {
				allErrs = append(allErrs, field.Invalid(field.NewPath("involvedObject", "namespace"), event.InvolvedObject.Namespace, "does not match event.namespace"))
			}
		}

		// For kinds we recognize, make sure involvedObject.Namespace is set for namespaced kinds
		if namespaced, err := isNamespacedKind(event.InvolvedObject.Kind, event.InvolvedObject.APIVersion); err == nil {
			if namespaced && len(event.InvolvedObject.Namespace) == 0 {
				allErrs = append(allErrs, field.Required(field.NewPath("involvedObject", "namespace"), fmt.Sprintf("required for kind %s", event.InvolvedObject.Kind)))
			}
			if !namespaced && len(event.InvolvedObject.Namespace) > 0 {
				allErrs = append(allErrs, field.Invalid(field.NewPath("involvedObject", "namespace"), event.InvolvedObject.Namespace, fmt.Sprintf("not allowed for kind %s", event.InvolvedObject.Kind)))
			}
		}

		for _, msg := range validation.IsDNS1123Subdomain(event.Namespace) {
			allErrs = append(allErrs, field.Invalid(field.NewPath("namespace"), event.Namespace, msg))
		}
	} else {
		// Make sure that Event's Namespace matches Object's Namespace, or is a SystemNamespace
		if event.Object != nil {
			if event.Object.Namespace != event.Namespace {
				allErrs = append(allErrs, field.Invalid(field.NewPath("object", "namespace"), event.Object.Namespace, "does not match event.namespace"))
			}
		} else if event.Namespace != metav1.NamespaceSystem {
			allErrs = append(allErrs, field.Invalid(field.NewPath("namespace"), event.Namespace, fmt.Sprintf("is not %v despite the lack of event.Object", metav1.NamespaceSystem)))
		}

		// Make sure that Origin is sane
		for _, msg := range validation.IsQualifiedName(event.Origin.Component) {
			allErrs = append(allErrs, field.Invalid(field.NewPath("origin", "component"), event.Origin.Component, msg))
		}

		for _, msg := range validation.IsCIdentifier(event.Action.Action) {
			allErrs = append(allErrs, field.Invalid(field.NewPath("origin", "action", "action"), event.Action.Action, msg))
		}

		if event.Severity == "" {
			allErrs = append(allErrs, field.Required(field.NewPath("severity"), "is required"))
		}

		// If Series field is set we need to make sure it's sane
		if event.Series != nil {
			if event.Series.UID == "" {
				allErrs = append(allErrs, field.Required(field.NewPath("series", "uid"), "is required"))
			}
			if event.Series.Count == 0 {
				allErrs = append(allErrs, field.Required(field.NewPath("series", "count"), "is required"))
			}
			if event.Series.HeartbeatTime.Time == zeroTime {
				allErrs = append(allErrs, field.Required(field.NewPath("series", "heartbeatTime"), "is required"))
			}
			if event.Series.LastObservedTime.Time == zeroTime {
				allErrs = append(allErrs, field.Required(field.NewPath("series", "lastObservedTime"), "is required"))
			}
			if event.Series.State == "" {
				allErrs = append(allErrs, field.Required(field.NewPath("series", "state"), "is required"))
			}
		}
	}
	return allErrs
}

func ValidateEventUpdate(oldEvent *api.Event, newEvent *api.Event) field.ErrorList {
	zeroTime := time.Time{}
	allErrs := ValidateEvent(newEvent)
	if (newEvent.EventTime.Time == zeroTime) != (oldEvent.EventTime.Time == zeroTime) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("eventTime"), newEvent.EventTime.Time, "update can't change old Event to the old one, or vice versa"))
		return allErrs
	}
	// "Old" Events does not set EventTime.
	if newEvent.EventTime.Time != zeroTime && oldEvent.EventTime.Time != zeroTime {
		if newEvent.Series == nil {
			allErrs = append(allErrs, field.Invalid(field.NewPath("series"), nil, " can't be set to nil during update"))
		} else {
			if oldEvent.Series != nil {
				if oldEvent.Series.Count > newEvent.Series.Count {
					allErrs = append(allErrs, field.Invalid(field.NewPath("series.count"), newEvent.Series.Count, " can't be decremented during update."))
				}
				if oldEvent.Series.HeartbeatTime.Time.After(newEvent.Series.HeartbeatTime.Time) {
					allErrs = append(allErrs, field.Invalid(field.NewPath("series.heartbeatTime"), newEvent.Series.HeartbeatTime, " can't be decremented during update."))
				}
				if oldEvent.Series.LastObservedTime.Time.After(newEvent.Series.LastObservedTime.Time) {
					allErrs = append(allErrs, field.Invalid(field.NewPath("series.lastObservedTime"), newEvent.Series.LastObservedTime.Time, " can't be decremented during update."))
				}
			}
		}
	}

	return allErrs
}

// Check whether the kind in groupVersion is scoped at the root of the api hierarchy
func isNamespacedKind(kind, groupVersion string) (bool, error) {
	group := apiutil.GetGroup(groupVersion)
	g, err := api.Registry.Group(group)
	if err != nil {
		return false, err
	}
	restMapping, err := g.RESTMapper.RESTMapping(schema.GroupKind{Group: group, Kind: kind}, apiutil.GetVersion(groupVersion))
	if err != nil {
		return false, err
	}
	scopeName := restMapping.Scope.Name()
	if scopeName == meta.RESTScopeNameNamespace {
		return true, nil
	}
	return false, nil
}
