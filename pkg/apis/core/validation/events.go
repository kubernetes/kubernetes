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
	"reflect"
	"time"

	v1 "k8s.io/api/core/v1"
	eventsv1beta1 "k8s.io/api/events/v1beta1"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/core"
)

const (
	ReportingInstanceLengthLimit = 128
	ActionLengthLimit            = 128
	ReasonLengthLimit            = 128
	NoteLengthLimit              = 1024
)

func ValidateEventCreate(event *core.Event, requestVersion schema.GroupVersion) field.ErrorList {
	// Make sure events always pass legacy validation.
	allErrs := legacyValidateEvent(event)
	if requestVersion == v1.SchemeGroupVersion || requestVersion == eventsv1beta1.SchemeGroupVersion {
		// No further validation for backwards compatibility.
		return allErrs
	}

	// Strict validation applies to creation via events.k8s.io/v1 API and newer.
	allErrs = append(allErrs, ValidateObjectMeta(&event.ObjectMeta, true, apimachineryvalidation.NameIsDNSSubdomain, field.NewPath("metadata"))...)
	allErrs = append(allErrs, validateV1EventSeries(event)...)
	zeroTime := time.Time{}
	if event.EventTime.Time == zeroTime {
		allErrs = append(allErrs, field.Required(field.NewPath("eventTime"), ""))
	}
	if event.Type != v1.EventTypeNormal && event.Type != v1.EventTypeWarning {
		allErrs = append(allErrs, field.Invalid(field.NewPath("type"), "", fmt.Sprintf("has invalid value: %v", event.Type)))
	}
	if event.FirstTimestamp.Time != zeroTime {
		allErrs = append(allErrs, field.Invalid(field.NewPath("firstTimestamp"), "", "needs to be unset"))
	}
	if event.LastTimestamp.Time != zeroTime {
		allErrs = append(allErrs, field.Invalid(field.NewPath("lastTimestamp"), "", "needs to be unset"))
	}
	if event.Count != 0 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("count"), "", "needs to be unset"))
	}
	if event.Source.Component != "" || event.Source.Host != "" {
		allErrs = append(allErrs, field.Invalid(field.NewPath("source"), "", "needs to be unset"))
	}
	return allErrs
}

func ValidateEventUpdate(newEvent, oldEvent *core.Event, requestVersion schema.GroupVersion) field.ErrorList {
	// Make sure the new event always passes legacy validation.
	allErrs := legacyValidateEvent(newEvent)
	if requestVersion == v1.SchemeGroupVersion || requestVersion == eventsv1beta1.SchemeGroupVersion {
		// No further validation for backwards compatibility.
		return allErrs
	}

	// Strict validation applies to update via events.k8s.io/v1 API and newer.
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&newEvent.ObjectMeta, &oldEvent.ObjectMeta, field.NewPath("metadata"))...)
	// if the series was modified, validate the new data
	if !reflect.DeepEqual(newEvent.Series, oldEvent.Series) {
		allErrs = append(allErrs, validateV1EventSeries(newEvent)...)
	}

	allErrs = append(allErrs, ValidateImmutableField(newEvent.InvolvedObject, oldEvent.InvolvedObject, field.NewPath("involvedObject"))...)
	allErrs = append(allErrs, ValidateImmutableField(newEvent.Reason, oldEvent.Reason, field.NewPath("reason"))...)
	allErrs = append(allErrs, ValidateImmutableField(newEvent.Message, oldEvent.Message, field.NewPath("message"))...)
	allErrs = append(allErrs, ValidateImmutableField(newEvent.Source, oldEvent.Source, field.NewPath("source"))...)
	allErrs = append(allErrs, ValidateImmutableField(newEvent.FirstTimestamp, oldEvent.FirstTimestamp, field.NewPath("firstTimestamp"))...)
	allErrs = append(allErrs, ValidateImmutableField(newEvent.LastTimestamp, oldEvent.LastTimestamp, field.NewPath("lastTimestamp"))...)
	allErrs = append(allErrs, ValidateImmutableField(newEvent.Count, oldEvent.Count, field.NewPath("count"))...)
	allErrs = append(allErrs, ValidateImmutableField(newEvent.Reason, oldEvent.Reason, field.NewPath("reason"))...)
	allErrs = append(allErrs, ValidateImmutableField(newEvent.Type, oldEvent.Type, field.NewPath("type"))...)
	allErrs = append(allErrs, ValidateImmutableField(newEvent.EventTime, oldEvent.EventTime, field.NewPath("eventTime"))...)
	allErrs = append(allErrs, ValidateImmutableField(newEvent.Action, oldEvent.Action, field.NewPath("action"))...)
	allErrs = append(allErrs, ValidateImmutableField(newEvent.Related, oldEvent.Related, field.NewPath("related"))...)
	allErrs = append(allErrs, ValidateImmutableField(newEvent.ReportingController, oldEvent.ReportingController, field.NewPath("reportingController"))...)
	allErrs = append(allErrs, ValidateImmutableField(newEvent.ReportingInstance, oldEvent.ReportingInstance, field.NewPath("reportingInstance"))...)

	return allErrs
}

func validateV1EventSeries(event *core.Event) field.ErrorList {
	allErrs := field.ErrorList{}
	zeroTime := time.Time{}
	if event.Series != nil {
		if event.Series.Count < 2 {
			allErrs = append(allErrs, field.Invalid(field.NewPath("series.count"), "", fmt.Sprintf("should be at least 2")))
		}
		if event.Series.LastObservedTime.Time == zeroTime {
			allErrs = append(allErrs, field.Required(field.NewPath("series.lastObservedTime"), ""))
		}
	}
	return allErrs
}

// legacyValidateEvent makes sure that the event makes sense.
func legacyValidateEvent(event *core.Event) field.ErrorList {
	allErrs := field.ErrorList{}
	// Because go
	zeroTime := time.Time{}

	// "New" Events need to have EventTime set, so it's validating old object.
	if event.EventTime.Time == zeroTime {
		// Make sure event.Namespace and the involvedInvolvedObject.Namespace agree
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

	} else {
		if len(event.InvolvedObject.Namespace) == 0 && event.Namespace != metav1.NamespaceDefault && event.Namespace != metav1.NamespaceSystem {
			allErrs = append(allErrs, field.Invalid(field.NewPath("involvedObject", "namespace"), event.InvolvedObject.Namespace, "does not match event.namespace"))
		}
		if len(event.ReportingController) == 0 {
			allErrs = append(allErrs, field.Required(field.NewPath("reportingController"), ""))
		}
		for _, msg := range validation.IsQualifiedName(event.ReportingController) {
			allErrs = append(allErrs, field.Invalid(field.NewPath("reportingController"), event.ReportingController, msg))
		}
		if len(event.ReportingInstance) == 0 {
			allErrs = append(allErrs, field.Required(field.NewPath("reportingInstance"), ""))
		}
		if len(event.ReportingInstance) > ReportingInstanceLengthLimit {
			allErrs = append(allErrs, field.Invalid(field.NewPath("reportingInstance"), "", fmt.Sprintf("can have at most %v characters", ReportingInstanceLengthLimit)))
		}
		if len(event.Action) == 0 {
			allErrs = append(allErrs, field.Required(field.NewPath("action"), ""))
		}
		if len(event.Action) > ActionLengthLimit {
			allErrs = append(allErrs, field.Invalid(field.NewPath("action"), "", fmt.Sprintf("can have at most %v characters", ActionLengthLimit)))
		}
		if len(event.Reason) == 0 {
			allErrs = append(allErrs, field.Required(field.NewPath("reason"), ""))
		}
		if len(event.Reason) > ReasonLengthLimit {
			allErrs = append(allErrs, field.Invalid(field.NewPath("reason"), "", fmt.Sprintf("can have at most %v characters", ReasonLengthLimit)))
		}
		if len(event.Message) > NoteLengthLimit {
			allErrs = append(allErrs, field.Invalid(field.NewPath("message"), "", fmt.Sprintf("can have at most %v characters", NoteLengthLimit)))
		}
	}

	for _, msg := range validation.IsDNS1123Subdomain(event.Namespace) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("namespace"), event.Namespace, msg))
	}
	return allErrs
}
