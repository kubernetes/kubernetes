/*
Copyright 2016 The Kubernetes Authors.

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
	"encoding/json"

	"k8s.io/apimachinery/pkg/util/validation/field"
	apivalidation "k8s.io/kubernetes/pkg/api/validation"
	pathvalidation "k8s.io/kubernetes/pkg/api/validation/path"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/controller/podautoscaler"
)

func ValidateScale(scale *autoscaling.Scale) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&scale.ObjectMeta, true, apivalidation.NameIsDNSSubdomain, field.NewPath("metadata"))...)

	if scale.Spec.Replicas < 0 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "replicas"), scale.Spec.Replicas, "must be greater than or equal to 0"))
	}

	return allErrs
}

// ValidateHorizontalPodAutoscaler can be used to check whether the given autoscaler name is valid.
// Prefix indicates this name will be used as part of generation, in which case trailing dashes are allowed.
var ValidateHorizontalPodAutoscalerName = apivalidation.ValidateReplicationControllerName

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
	if autoscaler.TargetCPUUtilizationPercentage != nil && *autoscaler.TargetCPUUtilizationPercentage < 1 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("targetCPUUtilizationPercentage"), autoscaler.TargetCPUUtilizationPercentage, "must be greater than 0"))
	}
	if refErrs := ValidateCrossVersionObjectReference(autoscaler.ScaleTargetRef, fldPath.Child("scaleTargetRef")); len(refErrs) > 0 {
		allErrs = append(allErrs, refErrs...)
	}
	return allErrs
}

func ValidateCrossVersionObjectReference(ref autoscaling.CrossVersionObjectReference, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(ref.Kind) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("kind"), ""))
	} else {
		for _, msg := range pathvalidation.IsValidPathSegmentName(ref.Kind) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("kind"), ref.Kind, msg))
		}
	}

	if len(ref.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
	} else {
		for _, msg := range pathvalidation.IsValidPathSegmentName(ref.Name) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), ref.Name, msg))
		}
	}

	return allErrs
}

func validateHorizontalPodAutoscalerAnnotations(annotations map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if annotationValue, found := annotations[podautoscaler.HpaCustomMetricsTargetAnnotationName]; found {
		// Try to parse the annotation
		var targetList v1beta1.CustomMetricTargetList
		if err := json.Unmarshal([]byte(annotationValue), &targetList); err != nil {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("annotations"), annotations, "failed to parse custom metrics target annotation"))
		} else {
			if len(targetList.Items) == 0 {
				allErrs = append(allErrs, field.Required(fldPath.Child("annotations", "items"), "custom metrics target must not be empty"))
			}
			for _, target := range targetList.Items {
				if target.Name == "" {
					allErrs = append(allErrs, field.Required(fldPath.Child("annotations", "items", "name"), "missing custom metric target name"))
				}
				if target.TargetValue.MilliValue() <= 0 {
					allErrs = append(allErrs, field.Invalid(fldPath.Child("annotations", "items", "value"), target.TargetValue, "custom metric target value must be greater than 0"))
				}
			}
		}
	}
	return allErrs
}

func ValidateHorizontalPodAutoscaler(autoscaler *autoscaling.HorizontalPodAutoscaler) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&autoscaler.ObjectMeta, true, ValidateHorizontalPodAutoscalerName, field.NewPath("metadata"))
	allErrs = append(allErrs, validateHorizontalPodAutoscalerSpec(autoscaler.Spec, field.NewPath("spec"))...)
	allErrs = append(allErrs, validateHorizontalPodAutoscalerAnnotations(autoscaler.Annotations, field.NewPath("metadata"))...)
	return allErrs
}

func ValidateHorizontalPodAutoscalerUpdate(newAutoscaler, oldAutoscaler *autoscaling.HorizontalPodAutoscaler) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&newAutoscaler.ObjectMeta, &oldAutoscaler.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, validateHorizontalPodAutoscalerSpec(newAutoscaler.Spec, field.NewPath("spec"))...)
	return allErrs
}

func ValidateHorizontalPodAutoscalerStatusUpdate(newAutoscaler, oldAutoscaler *autoscaling.HorizontalPodAutoscaler) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&newAutoscaler.ObjectMeta, &oldAutoscaler.ObjectMeta, field.NewPath("metadata"))
	status := newAutoscaler.Status
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.CurrentReplicas), field.NewPath("status", "currentReplicas"))...)
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.DesiredReplicas), field.NewPath("status", "desiredReplicasa"))...)
	return allErrs
}
