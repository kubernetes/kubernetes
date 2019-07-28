/*
Copyright 2019 The Kubernetes Authors.

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

package schema

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

// validateStructuralCompleteness checks that every specified field or array in s is also specified
// outside of value validation.
func validateStructuralCompleteness(s *Structural, fldPath *field.Path) field.ErrorList {
	if s == nil {
		return nil
	}

	return validateValueValidationCompleteness(s.ValueValidation, s, fldPath, fldPath)
}

func validateValueValidationCompleteness(v *ValueValidation, s *Structural, sPath, vPath *field.Path) field.ErrorList {
	if v == nil {
		return nil
	}
	if s == nil {
		return field.ErrorList{field.Required(sPath, fmt.Sprintf("because it is defined in %s", vPath.String()))}
	}

	allErrs := field.ErrorList{}

	allErrs = append(allErrs, validateNestedValueValidationCompleteness(v.Not, s, sPath, vPath.Child("not"))...)
	for i := range v.AllOf {
		allErrs = append(allErrs, validateNestedValueValidationCompleteness(&v.AllOf[i], s, sPath, vPath.Child("allOf").Index(i))...)
	}
	for i := range v.AnyOf {
		allErrs = append(allErrs, validateNestedValueValidationCompleteness(&v.AnyOf[i], s, sPath, vPath.Child("anyOf").Index(i))...)
	}
	for i := range v.OneOf {
		allErrs = append(allErrs, validateNestedValueValidationCompleteness(&v.OneOf[i], s, sPath, vPath.Child("oneOf").Index(i))...)
	}

	return allErrs
}

func validateNestedValueValidationCompleteness(v *NestedValueValidation, s *Structural, sPath, vPath *field.Path) field.ErrorList {
	if v == nil {
		return nil
	}
	if s == nil {
		return field.ErrorList{field.Required(sPath, fmt.Sprintf("because it is defined in %s", vPath.String()))}
	}

	allErrs := field.ErrorList{}

	allErrs = append(allErrs, validateValueValidationCompleteness(&v.ValueValidation, s, sPath, vPath)...)
	allErrs = append(allErrs, validateNestedValueValidationCompleteness(v.Items, s.Items, sPath.Child("items"), vPath.Child("items"))...)
	for k, vFld := range v.Properties {
		if sFld, ok := s.Properties[k]; !ok {
			allErrs = append(allErrs, field.Required(sPath.Child("properties").Key(k), fmt.Sprintf("because it is defined in %s", vPath.Child("properties").Key(k))))
		} else {
			allErrs = append(allErrs, validateNestedValueValidationCompleteness(&vFld, &sFld, sPath.Child("properties").Key(k), vPath.Child("properties").Key(k))...)
		}
	}

	// don't check additionalProperties as this is not allowed (and checked during validation)

	return allErrs
}
