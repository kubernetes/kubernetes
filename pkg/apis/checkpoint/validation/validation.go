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
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/checkpoint"
)

// ValidatePodCheckpoint validates a PodCheckpoint.
func ValidatePodCheckpoint(pc *checkpoint.PodCheckpoint) field.ErrorList {
	allErrs := field.ErrorList{}

	if pc.Spec.SourcePodName == "" {
		allErrs = append(allErrs, field.Required(field.NewPath("spec", "sourcePodName"), "sourcePodName is required"))
	}

	return allErrs
}

// ValidatePodCheckpointUpdate validates a PodCheckpoint update.
func ValidatePodCheckpointUpdate(newPC, oldPC *checkpoint.PodCheckpoint) field.ErrorList {
	allErrs := field.ErrorList{}

	// sourcePodName is immutable
	if newPC.Spec.SourcePodName != oldPC.Spec.SourcePodName {
		allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "sourcePodName"), newPC.Spec.SourcePodName, "field is immutable"))
	}

	return allErrs
}
