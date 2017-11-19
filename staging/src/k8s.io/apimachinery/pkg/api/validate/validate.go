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
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// NameValidator validates that the provided name is valid for a given resource type.
// Not all resources have the same validation rules for names. Prefix is true
// if the name will have a value appended to it.  If the name is not valid,
// this returns a list of descriptions of individual characteristics of the
// value that were not valid.  Otherwise this returns an empty list or nil.
type NameValidator func(name string, prefix bool) []string

const isNegativeErrorMsg string = `must be greater than or equal to 0`

// NonNegative validates that given int64 is not negative.
func NonNegative(value int64, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if value < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, value, isNegativeErrorMsg))
	}
	return allErrs
}

// NonNegativeQuantity validates that a given Quantity is not negative.
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

const isNotPositiveErrorMsg string = `must be greater than 0`

// Positive validates that given int64 is positive.
func Positive(value int64, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if value <= 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, value, isNotPositiveErrorMsg))
	}
	return allErrs
}

// PositiveQuantity validates that a given Quantity is positive.
func PositiveQuantity(value resource.Quantity, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if value.Cmp(resource.Quantity{}) <= 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, value.String(), isNotPositiveErrorMsg))
	}
	return allErrs
}

// DNS1123Label validates that a name is a proper DNS label.
func DNS1123Label(value string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range validation.IsDNS1123Label(value) {
		allErrs = append(allErrs, field.Invalid(fldPath, value, msg))
	}
	return allErrs
}

// DNS1123Subdomain validates that a name is a proper DNS subdomain.
func DNS1123Subdomain(value string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range validation.IsDNS1123Subdomain(value) {
		allErrs = append(allErrs, field.Invalid(fldPath, value, msg))
	}
	return allErrs
}
