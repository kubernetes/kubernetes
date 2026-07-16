/*
Copyright The Kubernetes Authors.

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

// MaxTimeoutSeconds is the largest spec.timeoutSeconds a PodCheckpoint may
// request. It is an API sanity bound; the kubelet further clamps the value to
// its configured PodCheckpointTimeout ceiling.
const MaxTimeoutSeconds = 3600

func validatePodCheckpointSpec(spec *checkpoint.PodCheckpointSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if spec.SourcePod == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("sourcePod"), "sourcePod is required"))
	} else if spec.SourcePod.Name == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("sourcePod", "name"), "name is required"))
	} else {
		// sourcePod.name references a pod in the same namespace; it must be a
		// valid pod name. The UID is opaque and not validated beyond matching.
		for _, msg := range apimachineryvalidation.NameIsDNSSubdomain(spec.SourcePod.Name, false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("sourcePod", "name"), spec.SourcePod.Name, msg))
		}
	}
	if spec.SourcePod != nil && spec.SourcePod.UID != nil && len(*spec.SourcePod.UID) == 0 {
		// nil means "not pinned"; a set but empty UID can never match a live
		// pod and would make every checkpoint attempt fail.
		allErrs = append(allErrs, field.Invalid(fldPath.Child("sourcePod", "uid"), *spec.SourcePod.UID, "may not be empty when set"))
	}

	// timeoutSeconds is a pointer, so unset is expressed as nil; a set value of
	// 0 is invalid rather than an alias for unset.
	if spec.TimeoutSeconds != nil && (*spec.TimeoutSeconds < 1 || *spec.TimeoutSeconds > MaxTimeoutSeconds) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("timeoutSeconds"), *spec.TimeoutSeconds, fmt.Sprintf("must be between 1 and %d", MaxTimeoutSeconds)))
	}

	return allErrs
}

// ValidatePodCheckpointUpdate validates a PodCheckpoint update.
func ValidatePodCheckpointUpdate(newPC, oldPC *checkpoint.PodCheckpoint) field.ErrorList {
	allErrs := apimachineryvalidation.ValidateObjectMetaUpdate(&newPC.ObjectMeta, &oldPC.ObjectMeta, field.NewPath("metadata"))

	// sourcePod is immutable; the referenced instance (name and pinned UID)
	// cannot be swapped after the PodCheckpoint is admitted.
	allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(newPC.Spec.SourcePod, oldPC.Spec.SourcePod, field.NewPath("spec", "sourcePod"))...)
	// Options are invocation input. Changing them after the asynchronous
	// operation has started would make retries observe different requests.
	allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(newPC.Spec.CheckpointOptions, oldPC.Spec.CheckpointOptions, field.NewPath("spec", "checkpointOptions"))...)

	allErrs = append(allErrs, validatePodCheckpointSpec(&newPC.Spec, field.NewPath("spec"))...)

	return allErrs
}

// ValidatePodCheckpointStatusUpdate validates a status update of a PodCheckpoint.
func ValidatePodCheckpointStatusUpdate(newPC, oldPC *checkpoint.PodCheckpoint) field.ErrorList {
	allErrs := apimachineryvalidation.ValidateObjectMetaUpdate(&newPC.ObjectMeta, &oldPC.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, metav1validation.ValidateConditions(newPC.Status.Conditions, field.NewPath("status", "conditions"))...)
	return allErrs
}
