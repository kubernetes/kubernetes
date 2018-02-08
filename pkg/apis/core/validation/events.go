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
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/core"
)

const (
	ReportingInstanceLengthLimit = 128
	ActionLengthLimit            = 128
	ReasonLengthLimit            = 128
	NoteLengthLimit              = 1024
)

// ValidateEvent makes sure that the event makes sense.
func ValidateEvent(event *core.Event) field.ErrorList {
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
		if len(event.InvolvedObject.Namespace) == 0 && event.Namespace != metav1.NamespaceSystem {
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
			allErrs = append(allErrs, field.Invalid(field.NewPath("repotingIntance"), "", fmt.Sprintf("can have at most %v characters", ReportingInstanceLengthLimit)))
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

	// For kinds we recognize, make sure InvolvedObject.Namespace is set for namespaced kinds
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
	return allErrs
}

// Check whether the kind in groupVersion is scoped at the root of the api hierarchy
func isNamespacedKind(kind, groupVersion string) (bool, error) {
	gv, err := schema.ParseGroupVersion(groupVersion)
	if err != nil {
		return false, err
	}
	g, err := legacyscheme.Registry.Group(gv.Group)
	if err != nil {
		return false, err
	}

	restMapping, err := g.RESTMapper.RESTMapping(schema.GroupKind{Group: gv.Group, Kind: kind}, gv.Version)
	if err != nil {
		return false, err
	}
	scopeName := restMapping.Scope.Name()
	if scopeName == meta.RESTScopeNameNamespace {
		return true, nil
	}
	return false, nil
}
