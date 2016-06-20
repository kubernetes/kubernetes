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

package validation

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/validation"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// ValidateEvent makes sure that the event makes sense.
func ValidateEvent(event *api.Event) field.ErrorList {
	allErrs := field.ErrorList{}
	// There is no namespace required for node or persistent volume.
	// However, older client code accidentally sets event.Namespace
	// to api.NamespaceDefault, so we accept that too, but "" is preferred.
	if (event.InvolvedObject.Kind == "Node" || event.InvolvedObject.Kind == "PersistentVolume") &&
		event.Namespace != api.NamespaceDefault &&
		event.Namespace != "" {
		allErrs = append(allErrs, field.Invalid(field.NewPath("involvedObject", "namespace"), event.InvolvedObject.Namespace, "not allowed for node"))
	}
	if event.InvolvedObject.Kind != "Node" &&
		event.InvolvedObject.Kind != "PersistentVolume" &&
		event.Namespace != event.InvolvedObject.Namespace {
		allErrs = append(allErrs, field.Invalid(field.NewPath("involvedObject", "namespace"), event.InvolvedObject.Namespace, "does not match involvedObject"))
	}
	for _, msg := range validation.IsDNS1123Subdomain(event.Namespace) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("namespace"), event.Namespace, msg))
	}
	return allErrs
}
