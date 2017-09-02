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
	unversionedvalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	apivalidation "k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/settings"
)

// ValidatePodPresetName can be used to check whether the given PodPreset name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidatePodPresetName(name string, prefix bool) []string {
	// TODO: Validate that there's name for the suffix inserted by the pods.
	// Currently this is just "-index". In the future we may allow a user
	// specified list of suffixes and we need  to validate the longest one.
	return apivalidation.NameIsDNSSubdomain(name, prefix)
}

// ValidatePodPresetSpec tests if required fields in the PodPreset spec are set.
func ValidatePodPresetSpec(spec *settings.PodPresetSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	allErrs = append(allErrs, unversionedvalidation.ValidateLabelSelector(&spec.Selector, fldPath.Child("selector"))...)

	if spec.Env == nil && spec.EnvFrom == nil && spec.VolumeMounts == nil && spec.Volumes == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("volumes", "env", "envFrom", "volumeMounts"), "must specify at least one"))
	}

	volumes, vErrs := apivalidation.ValidateVolumes(spec.Volumes, fldPath.Child("volumes"))
	allErrs = append(allErrs, vErrs...)
	allErrs = append(allErrs, apivalidation.ValidateEnv(spec.Env, fldPath.Child("env"))...)
	allErrs = append(allErrs, apivalidation.ValidateEnvFrom(spec.EnvFrom, fldPath.Child("envFrom"))...)
	allErrs = append(allErrs, apivalidation.ValidateVolumeMounts(spec.VolumeMounts, volumes, nil, fldPath.Child("volumeMounts"))...)

	return allErrs
}

// ValidatePodPreset validates a PodPreset.
func ValidatePodPreset(pip *settings.PodPreset) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&pip.ObjectMeta, true, ValidatePodPresetName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidatePodPresetSpec(&pip.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidatePodPresetUpdate tests if required fields in the PodPreset are set.
func ValidatePodPresetUpdate(pip, oldPip *settings.PodPreset) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&pip.ObjectMeta, &oldPip.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidatePodPresetSpec(&pip.Spec, field.NewPath("spec"))...)

	return allErrs
}
