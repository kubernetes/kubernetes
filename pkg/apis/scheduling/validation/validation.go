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
	"slices"
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

// validateCompositePodGroupName can be used to check whether the given
// name for a CompositePodGroup is valid.
var validateCompositePodGroupName = apimachineryvalidation.NameIsDNSSubdomain

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
	return apivalidation.ValidateObjectMetaUpdate(&podGroup.ObjectMeta, &oldPodGroup.ObjectMeta, field.NewPath("metadata"))
}

// ValidateWorkload tests if all fields in a Workload are set correctly.
func ValidateWorkload(workload *scheduling.Workload) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&workload.ObjectMeta, true, validateWorkloadName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateWorkloadSpec(&workload.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateWorkloadUpdate tests if an update to Workload is valid.
func ValidateWorkloadUpdate(workload, oldWorkload *scheduling.Workload) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&workload.ObjectMeta, &oldWorkload.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateWorkloadSpec(&workload.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateWorkloadSpec tests if the following conditions are satisfied:
// - All template names are unique.
// - Depth of the CompositePodGroupTemplates tree is not higher than 4.
// - All templates in the hierarchy share the same priority and PriorityClassName.
func ValidateWorkloadSpec(spec *scheduling.WorkloadSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, validateWorkloadTemplateNamesUniqueness(spec, fldPath)...)
	allErrs = append(allErrs, validateWorkloadTemplatesDepth(spec, fldPath.Child("compositePodGroupTemplates"))...)
	allErrs = append(allErrs, validateWorkloadPriority(spec, fldPath)...)
	return allErrs
}

func validateWorkloadTemplateNamesUniqueness(spec *scheduling.WorkloadSpec, fldPath *field.Path) field.ErrorList {
	if len(spec.PodGroupTemplates) > 0 {
		return nil
	}

	var allErrs field.ErrorList
	names := make(map[string]string)
	cpgListPath := fldPath.Child("compositePodGroupTemplates").String()
	for i, cpg := range spec.CompositePodGroupTemplates {
		cpgPath := fldPath.Child("compositePodGroupTemplates").Index(i)
		allErrs = append(allErrs, validateCompositePodGroupTemplateNames(&cpg, cpgPath, cpgListPath, names)...)
	}

	return allErrs
}

func validateCompositePodGroupTemplateNames(cpg *scheduling.CompositePodGroupTemplate, fldPath *field.Path, parentListPath string, names map[string]string) field.ErrorList {
	var allErrs field.ErrorList

	existingListPath, exists := names[cpg.Name]
	if !exists || existingListPath == parentListPath {
		names[cpg.Name] = parentListPath
	} else {
		allErrs = append(allErrs, field.Duplicate(fldPath.Child("name"), cpg.Name))
	}

	childCpgListPath := fldPath.Child("compositePodGroupTemplates").String()
	for i, child := range cpg.CompositePodGroupTemplates {
		childPath := fldPath.Child("compositePodGroupTemplates").Index(i)
		allErrs = append(allErrs, validateCompositePodGroupTemplateNames(&child, childPath, childCpgListPath, names)...)
	}

	childPgListPath := fldPath.Child("podGroupTemplates").String()
	for i, child := range cpg.PodGroupTemplates {
		childPath := fldPath.Child("podGroupTemplates").Index(i)
		existingListPath, exists := names[child.Name]
		if !exists || existingListPath == childPgListPath {
			names[child.Name] = childPgListPath
			continue
		}
		allErrs = append(allErrs, field.Duplicate(childPath.Child("name"), child.Name))
	}

	return allErrs
}

func validateWorkloadTemplatesDepth(spec *scheduling.WorkloadSpec, fldPath *field.Path) field.ErrorList {
	var walk func(*scheduling.CompositePodGroupTemplate, int) bool
	walk = func(cpg *scheduling.CompositePodGroupTemplate, depth int) bool {
		if depth == scheduling.WorkloadMaxTemplateDepth {
			return false
		}
		for _, child := range cpg.CompositePodGroupTemplates {
			if !walk(&child, depth+1) {
				return false
			}
		}
		return true
	}

	var allErrs field.ErrorList
	for i, cpg := range spec.CompositePodGroupTemplates {
		if !walk(&cpg, 1) {
			allErrs = append(allErrs, field.Invalid(fldPath.Index(i), cpg.CompositePodGroupTemplates, fmt.Sprintf("maximum template hierarchy depth is %d", scheduling.WorkloadMaxTemplateDepth)))
		}
	}
	return allErrs
}

func validateWorkloadPriority(spec *scheduling.WorkloadSpec, fldPath *field.Path) field.ErrorList {
	var firstClassName *string
	var firstPriority *int32
	var hasFirst bool

	checkPriority := func(className string, priority *int32) bool {
		if !hasFirst {
			firstClassName = &className
			firstPriority = priority
			hasFirst = true
			return true
		}
		if *firstClassName != className {
			return false
		}
		if (firstPriority == nil) != (priority == nil) {
			return false
		}
		if firstPriority != nil && priority != nil && *firstPriority != *priority {
			return false
		}
		return true
	}

	for _, pg := range spec.PodGroupTemplates {
		if !checkPriority(pg.PriorityClassName, pg.Priority) {
			return field.ErrorList{field.Invalid(fldPath, nil, "detected multiple priority configurations")}
		}
	}

	var checkCpg func(cpg *scheduling.CompositePodGroupTemplate) bool
	checkCpg = func(cpg *scheduling.CompositePodGroupTemplate) bool {
		if !checkPriority(cpg.PriorityClassName, cpg.Priority) {
			return false
		}
		for _, child := range cpg.CompositePodGroupTemplates {
			if !checkCpg(&child) {
				return false
			}
		}
		for _, child := range cpg.PodGroupTemplates {
			if !checkPriority(child.PriorityClassName, child.Priority) {
				return false
			}
		}
		return true
	}

	for _, cpg := range spec.CompositePodGroupTemplates {
		if !checkCpg(&cpg) {
			return field.ErrorList{field.Invalid(fldPath, nil, "detected multiple priority configurations")}
		}
	}

	return nil
}

// ValidatePodGroupStatusUpdate tests if an update to the status of a PodGroup is valid.
func ValidatePodGroupStatusUpdate(podGroup, oldPodGroup *scheduling.PodGroup) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&podGroup.ObjectMeta, &oldPodGroup.ObjectMeta, field.NewPath("metadata"))
	fldPath := field.NewPath("status")
	allErrs = append(allErrs, metav1validation.ValidateConditions(podGroup.Status.Conditions, fldPath.Child("conditions"))...)
	allErrs = append(allErrs, validatePodGroupResourceClaimStatuses(podGroup.Status.ResourceClaimStatuses, podGroup.Spec.ResourceClaims, fldPath.Child("resourceClaimStatuses"))...)
	return allErrs
}

func validatePodGroupResourceClaimStatuses(statuses []scheduling.PodGroupResourceClaimStatus, podGroupClaims []scheduling.PodGroupResourceClaim, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	for i, status := range statuses {
		idxPath := fldPath.Index(i)
		// There's no need to check the content of the name. If it matches an entry,
		// then it is valid, otherwise we reject it here.
		if !havePodGroupClaim(podGroupClaims, status.Name) {
			allErrs = append(allErrs, field.Invalid(idxPath.Child("name"), status.Name, "must match the name of an entry in `spec.resourceClaims`"))
		}
	}

	return allErrs
}

func havePodGroupClaim(podGroupClaims []scheduling.PodGroupResourceClaim, name string) bool {
	return slices.ContainsFunc(podGroupClaims, func(podGroupClaim scheduling.PodGroupResourceClaim) bool {
		return podGroupClaim.Name == name
	})
}

// ValidateCompositePodGroup tests if all fields in a CompositePodGroup are set correctly.
func ValidateCompositePodGroup(compositePodGroup *scheduling.CompositePodGroup) field.ErrorList {
	return apivalidation.ValidateObjectMeta(&compositePodGroup.ObjectMeta, true, validateCompositePodGroupName, field.NewPath("metadata"))
}

// ValidateCompositePodGroupUpdate tests if an update to CompositePodGroup is valid.
func ValidateCompositePodGroupUpdate(compositePodGroup, oldCompositePodGroup *scheduling.CompositePodGroup) field.ErrorList {
	return apivalidation.ValidateObjectMetaUpdate(&compositePodGroup.ObjectMeta, &oldCompositePodGroup.ObjectMeta, field.NewPath("metadata"))
}

// ValidateCompositePodGroupStatusUpdate tests if an update to the status of a CompositePodGroup is valid.
func ValidateCompositePodGroupStatusUpdate(compositePodGroup, oldCompositePodGroup *scheduling.CompositePodGroup) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&compositePodGroup.ObjectMeta, &oldCompositePodGroup.ObjectMeta, field.NewPath("metadata"))
	fldPath := field.NewPath("status")
	allErrs = append(allErrs, metav1validation.ValidateConditions(compositePodGroup.Status.Conditions, fldPath.Child("conditions"))...)
	return allErrs
}
