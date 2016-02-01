/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	apivalidation "k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// ValidateHorizontalPodAutoscaler can be used to check whether the given autoscaler name is valid.
// Prefix indicates this name will be used as part of generation, in which case trailing dashes are allowed.
func ValidateHorizontalPodAutoscalerName(name string, prefix bool) (bool, string) {
	// TODO: finally move it to pkg/api/validation and use nameIsDNSSubdomain function
	return apivalidation.ValidateReplicationControllerName(name, prefix)
}

func validateHorizontalPodAutoscalerSpec(autoscaler autoscaling.HorizontalPodAutoscalerSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if autoscaler.MinReplicas != nil && *autoscaler.MinReplicas < 1 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("minReplicas"), *autoscaler.MinReplicas, "must be greater than 0"))
	}
	if autoscaler.MaxReplicas < 1 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("maxReplicas"), autoscaler.MaxReplicas, "must be greater than 0"))
	}
	if autoscaler.MinReplicas != nil && autoscaler.MaxReplicas < *autoscaler.MinReplicas {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("maxReplicas"), autoscaler.MaxReplicas, "must be greater than or equal to `minReplicas`"))
	}
	if autoscaler.CPUUtilization != nil && autoscaler.CPUUtilization.TargetPercentage < 1 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("cpuUtilization", "targetPercentage"), autoscaler.CPUUtilization.TargetPercentage, "must be greater than 0"))
	}
	if refErrs := ValidateSubresourceReference(autoscaler.ScaleRef, fldPath.Child("scaleRef")); len(refErrs) > 0 {
		allErrs = append(allErrs, refErrs...)
	} else if autoscaler.ScaleRef.Subresource != "scale" {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("scaleRef", "subresource"), autoscaler.ScaleRef.Subresource, []string{"scale"}))
	}
	return allErrs
}

func ValidateSubresourceReference(ref autoscaling.SubresourceReference, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(ref.Kind) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("kind"), ""))
	} else if ok, msg := apivalidation.IsValidPathSegmentName(ref.Kind); !ok {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("kind"), ref.Kind, msg))
	}

	if len(ref.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
	} else if ok, msg := apivalidation.IsValidPathSegmentName(ref.Name); !ok {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), ref.Name, msg))
	}

	if len(ref.Subresource) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("subresource"), ""))
	} else if ok, msg := apivalidation.IsValidPathSegmentName(ref.Subresource); !ok {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("subresource"), ref.Subresource, msg))
	}
	return allErrs
}

func ValidateHorizontalPodAutoscaler(autoscaler *autoscaling.HorizontalPodAutoscaler) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&autoscaler.ObjectMeta, true, ValidateHorizontalPodAutoscalerName, field.NewPath("metadata"))
	allErrs = append(allErrs, validateHorizontalPodAutoscalerSpec(autoscaler.Spec, field.NewPath("spec"))...)
	return allErrs
}

func ValidateHorizontalPodAutoscalerUpdate(newAutoscaler, oldAutoscaler *autoscaling.HorizontalPodAutoscaler) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&newAutoscaler.ObjectMeta, &oldAutoscaler.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, validateHorizontalPodAutoscalerSpec(newAutoscaler.Spec, field.NewPath("spec"))...)
	return allErrs
}

func ValidateHorizontalPodAutoscalerStatusUpdate(controller, oldController *autoscaling.HorizontalPodAutoscaler) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&controller.ObjectMeta, &oldController.ObjectMeta, field.NewPath("metadata"))
	status := controller.Status
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.CurrentReplicas), field.NewPath("status", "currentReplicas"))...)
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.DesiredReplicas), field.NewPath("status", "desiredReplicasa"))...)
	return allErrs
}
