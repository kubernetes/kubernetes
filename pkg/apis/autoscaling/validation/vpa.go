/*
Copyright 2018 The Kubernetes Authors.

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
	metavalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	core "k8s.io/kubernetes/pkg/apis/core"
	corevalidation "k8s.io/kubernetes/pkg/apis/core/validation"
)

var supportedUpdateModes = sets.NewString(
	string(autoscaling.UpdateModeOff),
	string(autoscaling.UpdateModeInitial),
	string(autoscaling.UpdateModeAuto),
)

var supportedContainerScalingModes = sets.NewString(
	string(autoscaling.ContainerScalingModeAuto),
	string(autoscaling.ContainerScalingModeOff),
)

var supportedResources = sets.NewString(
	string(core.ResourceCPU),
	string(core.ResourceMemory),
)

func validateUpdateMode(updateMode *autoscaling.UpdateMode, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if updateMode != nil && !supportedUpdateModes.Has(string(*updateMode)) {
		allErrs = append(allErrs, field.NotSupported(fldPath, updateMode, supportedUpdateModes.List()))
	}
	return allErrs
}

func validateContainerScalingMode(containerScalingMode *autoscaling.ContainerScalingMode, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if containerScalingMode != nil && !supportedContainerScalingModes.Has(string(*containerScalingMode)) {
		allErrs = append(allErrs, field.NotSupported(fldPath, containerScalingMode, supportedContainerScalingModes.List()))
	}
	return allErrs
}

func validateResourceName(resourceName *core.ResourceName, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if resourceName != nil && !supportedResources.Has(string(*resourceName)) {
		allErrs = append(allErrs, field.NotSupported(fldPath, resourceName, supportedResources.List()))
	}
	return allErrs
}

func validatePodUpdatePolicy(podUpdatePolicy *autoscaling.PodUpdatePolicy, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if podUpdatePolicy != nil {
		allErrs = append(allErrs, validateUpdateMode(podUpdatePolicy.UpdateMode, fldPath.Child("updateMode"))...)
	}
	return allErrs
}

// Verifies that the core.ResourceList contains valid and supported resources (see supportedResources).
// Additionally checks that the quantity of resources in resourceList does not exceed the corresponding
// quantity in upperBound, if present.
func validateResourceList(resourceList core.ResourceList, upperBound core.ResourceList, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for resourceName, quantity := range resourceList {
		resPath := fldPath.Key(string(resourceName))
		// Validate resource name.
		allErrs = append(allErrs, validateResourceName(&resourceName, resPath)...)
		// Validate resource quantity.
		allErrs = append(allErrs, corevalidation.ValidateResourceQuantityValue(string(resourceName), quantity, resPath)...)
		if upperBound != nil {
			// Check that request <= limit.
			upperBoundQuantity, exists := upperBound[resourceName]
			if exists && quantity.Cmp(upperBoundQuantity) > 0 {
				allErrs = append(allErrs, field.Invalid(fldPath, quantity.String(),
					"must be less than or equal to the upper bound"))
			}
		}
	}
	return allErrs
}

func validateContainerResourcePolicy(containerResourcePolicy *autoscaling.ContainerResourcePolicy, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if containerResourcePolicy != nil {
		allErrs = append(allErrs, validateContainerScalingMode(containerResourcePolicy.Mode, fldPath.Child("mode"))...)
		allErrs = append(allErrs, validateResourceList(containerResourcePolicy.MinAllowed, containerResourcePolicy.MaxAllowed, fldPath.Child("minAllowed"))...)
		allErrs = append(allErrs, validateResourceList(containerResourcePolicy.MaxAllowed, core.ResourceList{}, fldPath.Child("maxAllowed"))...)
	}
	return allErrs
}

func validatePodResourcePolicy(podResourcePolicy *autoscaling.PodResourcePolicy, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if podResourcePolicy != nil {
		for i, containerPolicy := range podResourcePolicy.ContainerPolicies {
			allErrs = append(allErrs, validateContainerResourcePolicy(&containerPolicy, fldPath.Child("containerPolicies").Index(i))...)
		}
	}
	return allErrs
}

func validateVerticalPodAutoscalerSpec(spec *autoscaling.VerticalPodAutoscalerSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if spec.Selector == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("selector"), ""))
	} else {
		allErrs = append(allErrs, metavalidation.ValidateLabelSelector(spec.Selector, fldPath.Child("selector"))...)
	}
	allErrs = append(allErrs, validatePodUpdatePolicy(spec.UpdatePolicy, fldPath.Child("updatePolicy"))...)
	allErrs = append(allErrs, validatePodResourcePolicy(spec.ResourcePolicy, fldPath.Child("resourcePolicy"))...)
	return allErrs
}

func validateRecommendedContainerResources(recommendedContainerResources *autoscaling.RecommendedContainerResources, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if recommendedContainerResources != nil {
		allErrs = append(allErrs, validateResourceList(recommendedContainerResources.LowerBound, recommendedContainerResources.Target, fldPath.Child("minRecommended"))...)
		allErrs = append(allErrs, validateResourceList(recommendedContainerResources.Target, recommendedContainerResources.UpperBound, fldPath.Child("target"))...)
		allErrs = append(allErrs, validateResourceList(recommendedContainerResources.UpperBound, core.ResourceList{}, fldPath.Child("maxRecommended"))...)
	}
	return allErrs
}

func validateRecommendedPodResources(recommendedPodResources *autoscaling.RecommendedPodResources, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if recommendedPodResources != nil {
		for i, containerRecommendation := range recommendedPodResources.ContainerRecommendations {
			allErrs = append(allErrs, validateRecommendedContainerResources(&containerRecommendation, fldPath.Child("containerRecommendations").Index(i))...)
		}
	}
	return allErrs
}

func validateVerticalPodAutoscalerStatus(status *autoscaling.VerticalPodAutoscalerStatus, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if status != nil {
		allErrs = append(allErrs, validateRecommendedPodResources(status.Recommendation, fldPath.Child("recommendation"))...)
	}
	return allErrs
}

// ValidateVerticalPodAutoscalerName verifies that the vertical pod autoscaler name is valid.
var ValidateVerticalPodAutoscalerName = corevalidation.ValidateReplicationControllerName

// ValidateVerticalPodAutoscaler that VerticalPodAutoscaler is valid.
func ValidateVerticalPodAutoscaler(autoscaler *autoscaling.VerticalPodAutoscaler) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMeta(&autoscaler.ObjectMeta, true, ValidateVerticalPodAutoscalerName, field.NewPath("metadata"))
	if autoscaler != nil {
		allErrs = append(allErrs, validateVerticalPodAutoscalerSpec(&autoscaler.Spec, field.NewPath("spec"))...)
		allErrs = append(allErrs, validateVerticalPodAutoscalerStatus(&autoscaler.Status, field.NewPath("status"))...)
	}
	return allErrs
}

// ValidateVerticalPodAutoscalerUpdate that VerticalPodAutoscaler update is valid.
func ValidateVerticalPodAutoscalerUpdate(newAutoscaler, oldAutoscaler *autoscaling.VerticalPodAutoscaler) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&newAutoscaler.ObjectMeta, &oldAutoscaler.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, validateVerticalPodAutoscalerSpec(&newAutoscaler.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateVerticalPodAutoscalerStatusUpdate that VerticalPodAutoscaler status update is valid.
func ValidateVerticalPodAutoscalerStatusUpdate(newAutoscaler, oldAutoscaler *autoscaling.VerticalPodAutoscaler) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&newAutoscaler.ObjectMeta, &oldAutoscaler.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, validateVerticalPodAutoscalerStatus(&newAutoscaler.Status, field.NewPath("status"))...)
	return allErrs
}
