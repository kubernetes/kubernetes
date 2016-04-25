/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"reflect"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	unversionedvalidation "k8s.io/kubernetes/pkg/api/unversioned/validation"
	apivalidation "k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// ValidatePetSetName can be used to check whether the given PetSet
// name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidatePetSetName(name string, prefix bool) (bool, string) {
	// TODO: Validate that there's name for the suffix inserted by the pets.
	// Currently this is just "-index". In the future we may allow a user
	// specified list of suffixes and we need  to validate the longest one.
	return apivalidation.NameIsDNSSubdomain(name, prefix)
}

// Validates the given template and ensures that it is in accordance with the desired selector.
func ValidatePodTemplateSpecForPetSet(template *api.PodTemplateSpec, selector labels.Selector, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if template == nil {
		allErrs = append(allErrs, field.Required(fldPath, ""))
	} else {
		if !selector.Empty() {
			// Verify that the PetSet selector matches the labels in template.
			labels := labels.Set(template.Labels)
			if !selector.Matches(labels) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("metadata", "labels"), template.Labels, "`selector` does not match template `labels`"))
			}
		}
		// TODO: Add validation for PodSpec, currently this will check volumes, which we know will
		// fail. We should really check that the union of the given volumes and volumeClaims match
		// volume mounts in the containers.
		// allErrs = append(allErrs, apivalidation.ValidatePodTemplateSpec(template, fldPath)...)
		allErrs = append(allErrs, apivalidation.ValidateLabels(template.Labels, fldPath.Child("labels"))...)
		allErrs = append(allErrs, apivalidation.ValidateAnnotations(template.Annotations, fldPath.Child("annotations"))...)
		allErrs = append(allErrs, apivalidation.ValidatePodSpecificAnnotations(template.Annotations, fldPath.Child("annotations"))...)
	}
	return allErrs
}

// ValidatePetSetSpec tests if required fields in the PetSet spec are set.
func ValidatePetSetSpec(spec *apps.PetSetSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(spec.Replicas), fldPath.Child("replicas"))...)
	if spec.Selector == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("selector"), ""))
	} else {
		allErrs = append(allErrs, unversionedvalidation.ValidateLabelSelector(spec.Selector, fldPath.Child("selector"))...)
		if len(spec.Selector.MatchLabels)+len(spec.Selector.MatchExpressions) == 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("selector"), spec.Selector, "empty selector is not valid for petset."))
		}
	}

	selector, err := unversioned.LabelSelectorAsSelector(spec.Selector)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("selector"), spec.Selector, ""))
	} else {
		allErrs = append(allErrs, ValidatePodTemplateSpecForPetSet(&spec.Template, selector, fldPath.Child("template"))...)
	}

	if spec.Template.Spec.RestartPolicy != api.RestartPolicyAlways {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("template", "spec", "restartPolicy"), spec.Template.Spec.RestartPolicy, []string{string(api.RestartPolicyAlways)}))
	}

	return allErrs
}

// ValidatePetSet validates a PetSet.
func ValidatePetSet(petSet *apps.PetSet) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&petSet.ObjectMeta, true, ValidatePetSetName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidatePetSetSpec(&petSet.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidatePetSetUpdate tests if required fields in the PetSet are set.
func ValidatePetSetUpdate(petSet, oldPetSet *apps.PetSet) field.ErrorList {
	allErrs := field.ErrorList{}

	// TODO: For now we're taking the safe route and disallowing all updates to spec except for Spec.Replicas.
	// Enable on a case by case basis.
	restoreReplicas := petSet.Spec.Replicas
	petSet.Spec.Replicas = oldPetSet.Spec.Replicas

	// The generation changes for this update
	restoreGeneration := petSet.Generation
	petSet.Generation = oldPetSet.Generation

	if !reflect.DeepEqual(petSet, oldPetSet) {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("spec"), "updates to petset spec for fields other than 'replicas' are forbidden."))
	}
	petSet.Spec.Replicas = restoreReplicas
	petSet.Generation = restoreGeneration
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(petSet.Spec.Replicas), field.NewPath("spec", "replicas"))...)
	return allErrs
}

// ValidatePetSetStatusUpdate tests if required fields in the PetSet are set.
func ValidatePetSetStatusUpdate(petSet, oldPetSet *apps.PetSet) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&petSet.ObjectMeta, &oldPetSet.ObjectMeta, field.NewPath("metadata"))...)
	// TODO: Validate status.
	return allErrs
}
