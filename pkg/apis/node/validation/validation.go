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
	"k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/node"
)

// ValidateRuntimeClass validates a RuntimeClass.
func ValidateRuntimeClass(runtimeClass *node.RuntimeClass) field.ErrorList {
	allErrs := validation.ValidateObjectMeta(&runtimeClass.ObjectMeta, false, validation.NameIsDNSSubdomain, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateRuntimeClassSpec(&runtimeClass.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateRuntimeClassSpec validates spec of RuntimeClass.
func ValidateRuntimeClassSpec(spec *node.RuntimeClassSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	if spec.RuntimeHandler != "" {
		for _, msg := range validation.NameIsDNSSubdomain(spec.RuntimeHandler, false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("runtimeHandler"), spec.RuntimeHandler, msg))
		}
	}
	return allErrs
}

// ValidateRuntimeClassUpdate validates an update of RuntimeClass object.
func ValidateRuntimeClassUpdate(runtimeClass, oldRuntimeClass *node.RuntimeClass) field.ErrorList {
	allErrs := validation.ValidateObjectMetaUpdate(&runtimeClass.ObjectMeta, &oldRuntimeClass.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateRuntimeClassSpecUpdate(&runtimeClass.Spec, &oldRuntimeClass.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateRuntimeClassUpdate validates an update of a RuntimeClassSpec.
func ValidateRuntimeClassSpecUpdate(spec, oldSpec *node.RuntimeClassSpec, fldPath *field.Path) field.ErrorList {
	// All RuntimeClassSpec fields are currently immutable.
	return validation.ValidateImmutableField(spec, oldSpec, fldPath)
}
