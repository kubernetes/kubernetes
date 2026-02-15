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

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/api/validate/content"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
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

// ValidatePodGroup tests if all fields in a PodGroup are set correctly.
func ValidatePodGroup(podGroup *scheduling.PodGroup) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&podGroup.ObjectMeta, true, apivalidation.ValidatePodGroupName, field.NewPath("metadata"))
	allErrs = append(allErrs, validatePodGroupSpec(&podGroup.Spec, field.NewPath("spec"))...)
	return allErrs
}

func validatePodGroupSpec(spec *scheduling.PodGroupSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if spec.PodGroupTemplateRef != nil {
		allErrs = append(allErrs, validatePodGroupTemplateRef(spec.PodGroupTemplateRef, fldPath.Child("podGroupTemplateRef"))...)
	}
	allErrs = append(allErrs, validatePodGroupSchedulingPolicy(&spec.SchedulingPolicy, fldPath.Child("schedulingPolicy"))...)
	return allErrs
}

func validatePodGroupTemplateRef(ref *scheduling.PodGroupTemplateReference, fldPath *field.Path) field.ErrorList {
	var allErrs = field.ErrorList{}
	if ref.WorkloadName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("workloadName"), ""))
	} else {
		for _, detail := range apivalidation.ValidatePodGroupName(ref.WorkloadName, false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("workloadName"), ref.WorkloadName, detail).WithOrigin("format=k8s-short-name"))
		}
	}
	if ref.PodGroupTemplateName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("podGroupTemplateName"), ""))
	} else {
		for _, detail := range apivalidation.ValidatePodGroupName(ref.PodGroupTemplateName, false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("podGroupTemplateName"), ref.PodGroupTemplateName, detail).WithOrigin("format=k8s-short-name"))
		}
	}
	return allErrs
}

// ValidatePodGroupUpdate tests if an update to PodGroup is valid.
func ValidatePodGroupUpdate(podGroup, oldPodGroup *scheduling.PodGroup) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&podGroup.ObjectMeta, &oldPodGroup.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, validatePodGroupSpecUpdate(&podGroup.Spec, &oldPodGroup.Spec, field.NewPath("spec"))...)
	return allErrs
}

func validatePodGroupSpecUpdate(spec, oldSpec *scheduling.PodGroupSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.PodGroupTemplateRef, oldSpec.PodGroupTemplateRef, fldPath.Child("podGroupTemplateRef")).WithOrigin("immutable")...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.SchedulingPolicy, oldSpec.SchedulingPolicy, fldPath.Child("schedulingPolicy")).WithOrigin("immutable")...)
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
		allErrs = append(allErrs, validateControllerRef(spec.ControllerRef, fldPath.Child("controllerRef"))...)
	}
	allErrs = append(allErrs, validatePodGroupTemplates(fldPath, spec, operation.Create)...)
	return allErrs
}

func validatePodGroupTemplates(fldPath *field.Path, spec *scheduling.WorkloadSpec, op operation.Type) field.ErrorList {
	var allErrs field.ErrorList
	existingPodGroups := sets.New[string]()
	podGroupsPath := fldPath.Child("podGroupTemplates")
	if len(spec.PodGroupTemplates) == 0 {
		allErrs = append(allErrs, field.Required(podGroupsPath, "must have at least one item").MarkCoveredByDeclarative())
	} else if len(spec.PodGroupTemplates) > scheduling.WorkloadMaxPodGroupTemplates {
		allErrs = append(allErrs, field.TooMany(podGroupsPath, len(spec.PodGroupTemplates), scheduling.WorkloadMaxPodGroupTemplates).WithOrigin("maxItems").MarkCoveredByDeclarative())
	} else if op != operation.Update {
		// spec.PodGroupTemplates is immutable after create.
		for i := range spec.PodGroupTemplates {
			allErrs = append(allErrs, validatePodGroupTemplate(&spec.PodGroupTemplates[i], podGroupsPath.Index(i), existingPodGroups)...)
		}
	}
	return allErrs
}

func validateControllerRef(ref *scheduling.TypedLocalObjectReference, fldPath *field.Path) field.ErrorList {
	var allErrs = field.ErrorList{}
	if ref.APIGroup != "" {
		for _, msg := range validation.IsDNS1123Subdomain(ref.APIGroup) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("apiGroup"), ref.APIGroup, msg).WithOrigin("format=k8s-long-name").MarkCoveredByDeclarative())
		}
	}
	if ref.Kind == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("kind"), "").MarkCoveredByDeclarative())
	} else {
		for _, msg := range content.IsPathSegmentName(ref.Kind) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("kind"), ref.Kind, msg).WithOrigin("format=k8s-path-segment-name").MarkCoveredByDeclarative())
		}
	}
	if ref.Name == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), "").MarkCoveredByDeclarative())
	} else {
		for _, msg := range content.IsPathSegmentName(ref.Name) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), ref.Name, msg).WithOrigin("format=k8s-path-segment-name").MarkCoveredByDeclarative())
		}
	}
	return allErrs
}

func validatePodGroupTemplate(podGroupTemplate *scheduling.PodGroupTemplate, fldPath *field.Path, existingPodGroupTemplates sets.Set[string]) field.ErrorList {
	var allErrs field.ErrorList
	// To match the declarative validation behavior, we return Required for empty string.
	// Declarative validation treats "" as "missing" via validate.RequiredValue()
	// and returns early before checking the format constraint.
	if podGroupTemplate.Name == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), "").MarkCoveredByDeclarative())
	} else {
		for _, detail := range apivalidation.ValidatePodGroupName(podGroupTemplate.Name, false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), podGroupTemplate.Name, detail).WithOrigin("format=k8s-short-name").MarkCoveredByDeclarative())
		}
	}
	if existingPodGroupTemplates.Has(podGroupTemplate.Name) {
		// MarkCoveredByDeclarative is not needed here because the duplicate check is done.
		allErrs = append(allErrs, field.Duplicate(fldPath, podGroupTemplate.Name).MarkCoveredByDeclarative())
	} else {
		existingPodGroupTemplates.Insert(podGroupTemplate.Name)
	}
	allErrs = append(allErrs, validatePodGroupSchedulingPolicy(&podGroupTemplate.SchedulingPolicy, fldPath.Child("schedulingPolicy"))...)
	return allErrs
}

func validatePodGroupSchedulingPolicy(policy *scheduling.PodGroupSchedulingPolicy, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	var setFields []string

	if policy.Basic != nil {
		setFields = append(setFields, "`basic`")
	}
	if policy.Gang != nil {
		setFields = append(setFields, "`gang`")
	}

	switch {
	case len(setFields) == 0:
		allErrs = append(allErrs, field.Invalid(fldPath, "", "must specify one of: `basic`, `gang`").WithOrigin("union").MarkCoveredByDeclarative())
	case len(setFields) > 1:
		allErrs = append(allErrs, field.Invalid(fldPath, fmt.Sprintf("{%s}", strings.Join(setFields, ", ")),
			"exactly one of `basic`, `gang` is required, but multiple fields are set").WithOrigin("union").MarkCoveredByDeclarative())
	case policy.Basic != nil:
		allErrs = append(allErrs, validateBasicSchedulingPolicy(policy.Basic, fldPath.Child("basic"))...)
	case policy.Gang != nil:
		allErrs = append(allErrs, validateGangSchedulingPolicy(policy.Gang, fldPath.Child("gang"))...)
	}

	return allErrs
}

func validateBasicSchedulingPolicy(policy *scheduling.BasicSchedulingPolicy, fldPath *field.Path) field.ErrorList {
	// BasicSchedulingPolicy has no fields.
	return nil
}

func validateGangSchedulingPolicy(policy *scheduling.GangSchedulingPolicy, fldPath *field.Path) field.ErrorList {
	// To match the declarative validation behavior, we return Required for 0.
	// Declarative validation treats 0 as "missing" via validate.RequiredValue()
	// and returns early before checking the minimum constraint.
	// For non-zero values, declarative validation returns early without any validation,
	// so we don't mark them as covered.
	var allErrs field.ErrorList
	if policy.MinCount == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("minCount"), "").MarkCoveredByDeclarative())
	} else if policy.MinCount < 0 {
		allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(policy.MinCount), fldPath.Child("minCount")).WithOrigin("minimum").MarkCoveredByDeclarative()...)
	}
	return allErrs
}

// ValidateWorkloadUpdate tests if an update to Workload is valid.
func ValidateWorkloadUpdate(workload, oldWorkload *scheduling.Workload) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&workload.ObjectMeta, &oldWorkload.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, validateWorkloadSpecUpdate(&workload.Spec, &oldWorkload.Spec, field.NewPath("spec"))...)
	return allErrs
}

func validateWorkloadSpecUpdate(spec, oldSpec *scheduling.WorkloadSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if oldSpec.ControllerRef != nil {
		allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(spec.ControllerRef, oldSpec.ControllerRef, fldPath.Child("controllerRef")).WithOrigin("update").MarkCoveredByDeclarative()...)
	} else if spec.ControllerRef != nil {
		allErrs = append(allErrs, validateControllerRef(spec.ControllerRef, fldPath.Child("controllerRef"))...)
	}
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.PodGroupTemplates, oldSpec.PodGroupTemplates, fldPath.Child("podGroupTemplates")).WithOrigin("immutable").MarkCoveredByDeclarative()...)
	allErrs = append(allErrs, validatePodGroupTemplates(fldPath, spec, operation.Update)...)
	return allErrs
}

// ValidatePodGroupStatusUpdate tests if an update to the status of a PodGroup is valid.
func ValidatePodGroupStatusUpdate(podGroup, oldPodGroup *scheduling.PodGroup) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&podGroup.ObjectMeta, &oldPodGroup.ObjectMeta, field.NewPath("metadata"))
	return allErrs
}
