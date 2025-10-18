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
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/core"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	schedulingapiv1 "k8s.io/kubernetes/pkg/apis/scheduling/v1"
)

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

// ValidateWorkload tests if all fields in a Workload are set correctly.
func ValidateWorkload(workload *scheduling.Workload) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&workload.ObjectMeta, true, apivalidation.ValidateWorkloadName, field.NewPath("metadata"))
	allErrs = append(allErrs, validateWorkloadSpec(&workload.Spec, field.NewPath("spec"))...)
	return allErrs
}

func validateWorkloadSpec(spec *scheduling.WorkloadSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if spec.ControllerRef != nil {
		allErrs = append(allErrs, validateObjectReference(spec.ControllerRef, fldPath.Child("controllerRef"))...)
	}
	existingPodGroups := sets.New[string]()
	podGroupsPath := fldPath.Child("podGroups")
	for i := range spec.PodGroups {
		allErrs = append(allErrs, validatePodGroup(&spec.PodGroups[i], podGroupsPath.Index(i), existingPodGroups)...)
	}
	return allErrs
}

func validateObjectReference(ref *core.ObjectReference, fldPath *field.Path) field.ErrorList {
	return nil
}

func validatePodGroup(podGroup *scheduling.PodGroup, fldPath *field.Path, existingPodGroups sets.Set[string]) field.ErrorList {
	var allErrs field.ErrorList
	for _, detail := range apivalidation.ValidatePodGroupName(podGroup.Name, false) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), podGroup.Name, detail))
	}
	if existingPodGroups.Has(podGroup.Name) {
		allErrs = append(allErrs, field.Duplicate(fldPath.Child("name"), podGroup.Name))
	} else {
		existingPodGroups.Insert(podGroup.Name)
	}
	if podGroup.Replicas == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("replicas"), "must be set"))
	} else {
		allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(*podGroup.Replicas), fldPath.Child("replicas"))...)
	}
	allErrs = append(allErrs, validatePodGroupPolicy(&podGroup.Policy, fldPath.Child("policy"))...)
	return allErrs
}

func validatePodGroupPolicy(policy *scheduling.PodGroupPolicy, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	switch policy.Kind {
	case scheduling.PodGroupPolicyKindDefault:
		if policy.Default != nil {
			allErrs = append(allErrs, validateDefaultSchedulingPolicy(policy.Default, fldPath.Child("default"))...)
		} else {
			allErrs = append(allErrs, field.Required(fldPath.Child("default"), "must be specified when kind is set to Default"))
		}
		if policy.Gang != nil {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("gang"), "should not be specified when kind is not set to Gang"))
		}
	case scheduling.PodGroupPolicyKindGang:
		if policy.Gang != nil {
			allErrs = append(allErrs, validateGangSchedulingPolicy(policy.Gang, fldPath.Child("gang"))...)
		} else {
			allErrs = append(allErrs, field.Required(fldPath.Child("gang"), "must be specified when kind is set to Gang"))
		}
		if policy.Default != nil {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("default"), "should not be specified when kind is not set to Default"))
		}
	default:
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("kind"), policy.Kind, []scheduling.PodGroupPolicyKind{scheduling.PodGroupPolicyKindDefault, scheduling.PodGroupPolicyKindGang}))
	}
	return allErrs
}

func validateDefaultSchedulingPolicy(policy *scheduling.DefaultSchedulingPolicy, fldPath *field.Path) field.ErrorList {
	// DefaultSchedulingPolicy has no fields.
	return nil
}

func validateGangSchedulingPolicy(policy *scheduling.GangSchedulingPolicy, fldPath *field.Path) field.ErrorList {
	allErrs := apivalidation.ValidatePositiveField(int64(policy.MinCount), fldPath.Child("minCount"))
	return allErrs
}

// ValidateWorkloadUpdate tests if an update to Workload is valid.
func ValidateWorkloadUpdate(workload, oldWorkload *scheduling.Workload) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&workload.ObjectMeta, &oldWorkload.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, validateWorkloadSpec(&workload.Spec, field.NewPath("spec"))...)
	allErrs = append(allErrs, validateWorkloadSpecUpdate(&workload.Spec, &oldWorkload.Spec, field.NewPath("spec"))...)
	return allErrs
}

func validateWorkloadSpecUpdate(spec, oldSpec *scheduling.WorkloadSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if oldSpec.ControllerRef != nil {
		allErrs = apimachineryvalidation.ValidateImmutableField(spec.ControllerRef, oldSpec.ControllerRef, field.NewPath("controllerRef"))
	}
	podGroupsPath := fldPath.Child("podGroups")
	if len(spec.PodGroups) != len(oldSpec.PodGroups) {
		allErrs = append(allErrs, field.Forbidden(podGroupsPath, "number of items should not be modified"))
	}
	for i := 0; i < min(len(spec.PodGroups), len(oldSpec.PodGroups)); i++ {
		allErrs = append(allErrs, validatePodGroupUpdate(&spec.PodGroups[i], &oldSpec.PodGroups[i], podGroupsPath.Index(i))...)
	}
	return allErrs
}

func validatePodGroupUpdate(podGroup, oldPodGroup *scheduling.PodGroup, fldPath *field.Path) field.ErrorList {
	allErrs := apimachineryvalidation.ValidateImmutableField(podGroup.Name, oldPodGroup.Name, field.NewPath("name"))
	return allErrs
}
