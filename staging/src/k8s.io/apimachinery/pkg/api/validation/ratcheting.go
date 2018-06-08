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
	"k8s.io/apimachinery/pkg/util/validation/field"
)

type RatchetingValidators []RatchetingValidator

type RatchetingValidator func(data interface{}, fldPath *field.Path) field.ErrorList

// ValidateRatchetingCreate validates newData with each the given validators.
// newData must be of the type expected by the validation functions.
func ValidateRatchetingCreate(newData interface{}, fldPath *field.Path, validators []RatchetingValidator) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, validator := range validators {
		allErrs = append(allErrs, validator(newData, fldPath)...)
	}
	return allErrs
}

// ValidateRatchetingUpdate validates newData with the given validators, if existingData is successfully validated by that validator.
// This ensures that valid objects remain valid, but that updates to existing invalid objects are still permitted.
// newData and existingData must be of the type expected by the validation functions.
func ValidateRatchetingUpdate(newData, existingData interface{}, fldPath *field.Path, validators RatchetingValidators) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, validator := range validators {
		// only require the new data to pass this validation if the existing data also passes it
		if len(validator(existingData, fldPath)) == 0 {
			allErrs = append(allErrs, validator(newData, fldPath)...)
		}
	}
	return allErrs
}
