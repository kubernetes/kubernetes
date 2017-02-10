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
	"strings"

	pathvalidation "k8s.io/apimachinery/pkg/api/validation/path"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	apivalidation "k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
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
	if refErrs := ValidateCrossVersionObjectReference(autoscaler.ScaleTargetRef, fldPath.Child("scaleTargetRef")); len(refErrs) > 0 {
		allErrs = append(allErrs, refErrs...)
	}
	if refErrs := validateMetrics(autoscaler.Metrics, fldPath.Child("metrics")); len(refErrs) > 0 {
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

func ValidateHorizontalPodAutoscalerStatusUpdate(newAutoscaler, oldAutoscaler *autoscaling.HorizontalPodAutoscaler) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&newAutoscaler.ObjectMeta, &oldAutoscaler.ObjectMeta, field.NewPath("metadata"))
	status := newAutoscaler.Status
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.CurrentReplicas), field.NewPath("status", "currentReplicas"))...)
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.DesiredReplicas), field.NewPath("status", "desiredReplicasa"))...)
	return allErrs
}

func validateMetrics(metrics []autoscaling.MetricSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	for i, metricSpec := range metrics {
		idxPath := fldPath.Index(i)
		if targetErrs := validateMetricSpec(metricSpec, idxPath); len(targetErrs) > 0 {
			allErrs = append(allErrs, targetErrs...)
		}
	}

	return allErrs
}

var validMetricSourceTypes = sets.NewString(string(autoscaling.ObjectMetricSourceType), string(autoscaling.PodsMetricSourceType), string(autoscaling.ResourceMetricSourceType))
var validMetricSourceTypesList = validMetricSourceTypes.List()

func validateMetricSpec(spec autoscaling.MetricSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(string(spec.Type)) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("type"), "must specify a metric source type"))
	}

	if !validMetricSourceTypes.Has(string(spec.Type)) {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("type"), spec.Type, validMetricSourceTypesList))
	}

	typesPresent := sets.NewString()
	if spec.Object != nil {
		typesPresent.Insert("object")
		if typesPresent.Len() == 1 {
			allErrs = append(allErrs, validateObjectSource(spec.Object, fldPath.Child("object"))...)
		}
	}

	if spec.Pods != nil {
		typesPresent.Insert("pods")
		if typesPresent.Len() == 1 {
			allErrs = append(allErrs, validatePodsSource(spec.Pods, fldPath.Child("pods"))...)
		}
	}

	if spec.Resource != nil {
		typesPresent.Insert("resource")
		if typesPresent.Len() == 1 {
			allErrs = append(allErrs, validateResourceSource(spec.Resource, fldPath.Child("resource"))...)
		}
	}

	expectedField := strings.ToLower(string(spec.Type))

	if !typesPresent.Has(expectedField) {
		allErrs = append(allErrs, field.Required(fldPath.Child(expectedField), "must populate information for the given metric source"))
	}

	if typesPresent.Len() != 1 {
		typesPresent.Delete(expectedField)
		for typ := range typesPresent {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child(typ), "must populate the given metric source only"))
		}
	}

	return allErrs
}

func validateObjectSource(src *autoscaling.ObjectMetricSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	allErrs = append(allErrs, ValidateCrossVersionObjectReference(src.Target, fldPath.Child("target"))...)

	if len(src.MetricName) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("metricName"), "must specify a metric name"))
	}

	if src.TargetValue.Sign() != 1 {
		allErrs = append(allErrs, field.Required(fldPath.Child("targetValue"), "must specify a positive target value"))
	}

	return allErrs
}

func validatePodsSource(src *autoscaling.PodsMetricSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(src.MetricName) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("metricName"), "must specify a metric name"))
	}

	if src.TargetAverageValue.Sign() != 1 {
		allErrs = append(allErrs, field.Required(fldPath.Child("targetAverageValue"), "must specify a positive target value"))
	}

	return allErrs
}

func validateResourceSource(src *autoscaling.ResourceMetricSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(src.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), "must specify a resource name"))
	}

	if src.TargetAverageUtilization == nil && src.TargetAverageValue == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("targetAverageUtilization"), "must set either a target raw value or a target utilization"))
	}

	if src.TargetAverageUtilization != nil && *src.TargetAverageUtilization < 1 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("targetAverageUtilization"), src.TargetAverageUtilization, "must be greater than 0"))
	}

	if src.TargetAverageUtilization != nil && src.TargetAverageValue != nil {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("targetAverageValue"), "may not set both a target raw value and a target utilization"))
	}

	if src.TargetAverageValue != nil && src.TargetAverageValue.Sign() != 1 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("targetAverageValue"), src.TargetAverageValue, "must be positive"))
	}

	return allErrs
}
