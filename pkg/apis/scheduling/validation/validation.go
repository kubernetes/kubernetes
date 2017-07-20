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
	"k8s.io/apimachinery/pkg/util/validation/field"
	apivalidation "k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/scheduling"
)

// ValidatePriorityClassName can be used to check whether the given priority
// class name is valid.
var ValidatePriorityClassName = apivalidation.NameIsDNSSubdomain

// ValidatePriorityClass tests whether required fields in the PriorityClass are
// set correctly.
func ValidatePriorityClass(pc *scheduling.PriorityClass) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&pc.ObjectMeta, false, ValidatePriorityClassName, field.NewPath("metadata"))...)
	// The "Value" field can be any valid integer. So, no need to validate.
	return allErrs
}

// ValidatePriorityClassUpdate tests if required fields in the PriorityClass are
// set and are valid. PriorityClass does not allow updating Name, and Value.
func ValidatePriorityClassUpdate(pc, oldPc *scheduling.PriorityClass) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&pc.ObjectMeta, &oldPc.ObjectMeta, field.NewPath("metadata"))
	// Name is immutable and is checked by the ObjectMeta validator.
	if pc.Value != oldPc.Value {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("Value"), "may not be changed in an update."))
	}
	return allErrs
}
