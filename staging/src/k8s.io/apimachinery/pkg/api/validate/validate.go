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

package validate

import (
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

const isNegativeErrorMsg string = `must be greater than or equal to 0`

// NonNegative validates that given int64 is not negative.
func NonNegative(value int64, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if value < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, value, isNegativeErrorMsg))
	}
	return allErrs
}

// NonNegativeQuantity validates that a given Quantity is not negative
func NonNegativeQuantity(value resource.Quantity, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if value.Cmp(resource.Quantity{}) < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, value.String(), isNegativeErrorMsg))
	}
	return allErrs
}

// NonNegativeDuration validates that given Duration is not negative.
func NonNegativeDuration(value time.Duration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if int64(value) < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, value, `must be greater than or equal to 0`))
	}
	return allErrs
}
