/*
Copyright 2017 The Kubernetes Authors.

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
	"strings"

	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	schedulingapiv1 "k8s.io/kubernetes/pkg/apis/scheduling/v1"
)

// validateWorkloadName can be used to check whether the given
// name for a Workload is valid.
var validateWorkloadName = apimachineryvalidation.NameIsDNSSubdomain

// ValidatePriorityClass tests whether required fields in the PriorityClass are
// set correctly.
func ValidatePriorityClass(pc *scheduling.PriorityClass) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&pc.ObjectMeta, false, apimachineryvalidation.NameIsDNSSubdomain, field.NewPath("metadata"))...)
	// If the priorityClass starts with a system prefix, it must be one of the
	// predefined system priority classes.
	if strings.HasPrefix(pc.Name, scheduling.SystemPriorityClassPrefix) {
		if is, err := schedulingapiv1.IsKnownSystemPriorityClass(pc.Name, pc.Value, pc.GlobalDefault); !is {
			allErrs = append(allErrs, field.Forbidden(field.NewPath("metadata", "name"), "priority class names with '"+scheduling.SystemPriorityClassPrefix+"' prefix are reserved for system use only. error: "+err.Error()))
		}
	} else if pc.Value > scheduling.HighestUserDefinablePriority {
		// Non-system critical priority classes are not allowed to have a value larger than HighestUserDefinablePriority.
		allErrs = append(allErrs, field.Forbidden(field.NewPath("value"), fmt.Sprintf("maximum allowed value of a user defined priority is %v", scheduling.HighestUserDefinablePriority)))
	}
	if pc.PreemptionPolicy != nil {
		allErrs = append(allErrs, apivalidation.ValidatePreemptionPolicy(pc.PreemptionPolicy, field.NewPath("preemptionPolicy"))...)
	}
	return allErrs
}

// ValidatePriorityClassUpdate tests if required fields in the PriorityClass are
// set and are valid. PriorityClass does not allow updating name, value, and preemptionPolicy.
func ValidatePriorityClassUpdate(pc, oldPc *scheduling.PriorityClass) field.ErrorList {
	// name is immutable and is checked by the ObjectMeta validator.
	allErrs := apivalidation.ValidateObjectMetaUpdate(&pc.ObjectMeta, &oldPc.ObjectMeta, field.NewPath("metadata"))
	// value is immutable.
	if pc.Value != oldPc.Value {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("value"), "may not be changed in an update."))
	}
	// preemptionPolicy is immutable.
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(pc.PreemptionPolicy, oldPc.PreemptionPolicy, field.NewPath("preemptionPolicy"))...)
	return allErrs
}

// ValidatePodGroup tests if all fields in a PodGroup are set correctly.
func ValidatePodGroup(podGroup *scheduling.PodGroup) field.ErrorList {
	return apivalidation.ValidateObjectMeta(&podGroup.ObjectMeta, true, apivalidation.ValidatePodGroupName, field.NewPath("metadata"))
}

// ValidatePodGroupUpdate tests if an update to PodGroup is valid.
func ValidatePodGroupUpdate(podGroup, oldPodGroup *scheduling.PodGroup) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&podGroup.ObjectMeta, &oldPodGroup.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, validatePodGroupSpecUpdate(&podGroup.Spec, &oldPodGroup.Spec, field.NewPath("spec"))...)
	return allErrs
}

func validatePodGroupSpecUpdate(spec, oldSpec *scheduling.PodGroupSpec, fldPath *field.Path) field.ErrorList {
	allErrs := apivalidation.ValidateImmutableField(spec.PodGroupTemplateRef, oldSpec.PodGroupTemplateRef, fldPath.Child("podGroupTemplateRef")).WithOrigin("immutable").MarkCoveredByDeclarative()
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.SchedulingPolicy, oldSpec.SchedulingPolicy, fldPath.Child("schedulingPolicy")).WithOrigin("immutable").MarkCoveredByDeclarative()...)
	return allErrs
}

// ValidateWorkload tests if all fields in a Workload are set correctly.
func ValidateWorkload(workload *scheduling.Workload) field.ErrorList {
	return apivalidation.ValidateObjectMeta(&workload.ObjectMeta, true, validateWorkloadName, field.NewPath("metadata"))
}

// ValidateWorkloadUpdate tests if an update to Workload is valid.
func ValidateWorkloadUpdate(workload, oldWorkload *scheduling.Workload) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&workload.ObjectMeta, &oldWorkload.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, validateWorkloadSpecUpdate(&workload.Spec, &oldWorkload.Spec, field.NewPath("spec"))...)
	return allErrs
}

func validateWorkloadSpecUpdate(spec, oldSpec *scheduling.WorkloadSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(spec.ControllerRef, oldSpec.ControllerRef, fldPath.Child("controllerRef")).WithOrigin("immutable").MarkCoveredByDeclarative()...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.PodGroupTemplates, oldSpec.PodGroupTemplates, fldPath.Child("podGroupTemplates")).WithOrigin("immutable").MarkCoveredByDeclarative()...)
	return allErrs
}

// ValidatePodGroupStatusUpdate tests if an update to the status of a PodGroup is valid.
func ValidatePodGroupStatusUpdate(podGroup, oldPodGroup *scheduling.PodGroup) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&podGroup.ObjectMeta, &oldPodGroup.ObjectMeta, field.NewPath("metadata"))
	fldPath := field.NewPath("status")
	allErrs = append(allErrs, metav1validation.ValidateConditions(podGroup.Status.Conditions, fldPath.Child("conditions"))...)
	return allErrs
}
