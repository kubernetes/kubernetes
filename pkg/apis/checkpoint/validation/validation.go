/*
Copyright 2026 The Kubernetes Authors.

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
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/checkpoint"
)

// ValidatePodCheckpointName validates the name of a PodCheckpoint.
func ValidatePodCheckpointName(name string, prefix bool) []string {
	return apimachineryvalidation.NameIsDNSSubdomain(name, prefix)
}

// ValidatePodCheckpoint validates a PodCheckpoint.
func ValidatePodCheckpoint(pc *checkpoint.PodCheckpoint) field.ErrorList {
	allErrs := apimachineryvalidation.ValidateObjectMeta(&pc.ObjectMeta, true, ValidatePodCheckpointName, field.NewPath("metadata"))
	allErrs = append(allErrs, validatePodCheckpointSpec(&pc.Spec, field.NewPath("spec"))...)
	return allErrs
}

func validatePodCheckpointSpec(spec *checkpoint.PodCheckpointSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if spec.SourcePodName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("sourcePodName"), "sourcePodName is required"))
	} else {
		// sourcePodName references a pod in the same namespace; it must be a
		// valid pod name.
		for _, msg := range apimachineryvalidation.NameIsDNSSubdomain(spec.SourcePodName, false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("sourcePodName"), spec.SourcePodName, msg))
		}
	}

	if spec.TimeoutSeconds != nil && *spec.TimeoutSeconds < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("timeoutSeconds"), *spec.TimeoutSeconds, "must be greater than or equal to 0"))
	}

	return allErrs
}

// ValidatePodCheckpointUpdate validates a PodCheckpoint update.
func ValidatePodCheckpointUpdate(newPC, oldPC *checkpoint.PodCheckpoint) field.ErrorList {
	allErrs := apimachineryvalidation.ValidateObjectMetaUpdate(&newPC.ObjectMeta, &oldPC.ObjectMeta, field.NewPath("metadata"))

	// sourcePodName is immutable.
	allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(newPC.Spec.SourcePodName, oldPC.Spec.SourcePodName, field.NewPath("spec", "sourcePodName"))...)
	// sourcePodUID is immutable; the pinned instance cannot be swapped after the
	// PodCheckpoint is admitted.
	allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(newPC.Spec.SourcePodUID, oldPC.Spec.SourcePodUID, field.NewPath("spec", "sourcePodUID"))...)

	allErrs = append(allErrs, validatePodCheckpointSpec(&newPC.Spec, field.NewPath("spec"))...)

	return allErrs
}

// ValidatePodCheckpointStatusUpdate validates a status update of a PodCheckpoint.
func ValidatePodCheckpointStatusUpdate(newPC, oldPC *checkpoint.PodCheckpoint) field.ErrorList {
	allErrs := apimachineryvalidation.ValidateObjectMetaUpdate(&newPC.ObjectMeta, &oldPC.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, metav1validation.ValidateConditions(newPC.Status.Conditions, field.NewPath("status", "conditions"))...)
	return allErrs
}
