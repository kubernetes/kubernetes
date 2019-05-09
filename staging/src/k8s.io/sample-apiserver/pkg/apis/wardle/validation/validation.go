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
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/sample-apiserver/pkg/apis/wardle"
)

// ValidateFlunder validates a Flunder.
func ValidateFlunder(f *wardle.Flunder) field.ErrorList {
	allErrs := field.ErrorList{}

	allErrs = append(allErrs, ValidateFlunderSpec(&f.Spec, field.NewPath("spec"))...)

	return allErrs
}

// ValidateFlunderSpec validates a FlunderSpec.
func ValidateFlunderSpec(s *wardle.FlunderSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(s.FlunderReference) != 0 && len(s.FischerReference) != 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("fischerReference"), s.FischerReference, "cannot be set with flunderReference at the same time"))
	} else if len(s.FlunderReference) != 0 && s.ReferenceType != wardle.FlunderReferenceType {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("flunderReference"), s.FlunderReference, "cannot be set if referenceType is not Flunder"))
	} else if len(s.FischerReference) != 0 && s.ReferenceType != wardle.FischerReferenceType {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("fischerReference"), s.FischerReference, "cannot be set if referenceType is not Fischer"))
	} else if len(s.FischerReference) == 0 && s.ReferenceType == wardle.FischerReferenceType {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("fischerReference"), s.FischerReference, "cannot be empty if referenceType is Fischer"))
	} else if len(s.FlunderReference) == 0 && s.ReferenceType == wardle.FlunderReferenceType {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("flunderReference"), s.FlunderReference, "cannot be empty if referenceType is Flunder"))
	}

	if len(s.ReferenceType) != 0 && s.ReferenceType != wardle.FischerReferenceType && s.ReferenceType != wardle.FlunderReferenceType {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("referenceType"), s.ReferenceType, "must be Flunder or Fischer"))
	}

	return allErrs
}
