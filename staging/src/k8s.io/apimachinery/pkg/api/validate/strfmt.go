/*
Copyright 2025 The Kubernetes Authors.

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
	"context"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/api/validate/content"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

const (
	uuidErrorMessage = "must be a lowercase UUID in 8-4-4-4-12 format"
)

// ShortName verifies that the specified value is a valid "short name"
// (sometimes known as a "DNS label").
//   - must not be empty
//   - must be less than 64 characters long
//   - must start and end with lower-case alphanumeric characters
//   - must contain only lower-case alphanumeric characters or dashes
//
// All errors returned by this function will be "invalid" type errors. If the
// caller wants better errors, it must take responsibility for checking things
// like required/optional and max-length.
func ShortName[T ~string](_ context.Context, op operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value == nil {
		return nil
	}
	var allErrs field.ErrorList
	for _, msg := range content.IsDNS1123Label((string)(*value)) {
		allErrs = append(allErrs, field.Invalid(fldPath, *value, msg).WithOrigin("format=k8s-short-name"))
	}
	return allErrs
}

// LongName verifies that the specified value is a valid "long name"
// (sometimes known as a "DNS subdomain").
//   - must not be empty
//   - must be less than 254 characters long
//   - each element must start and end with lower-case alphanumeric characters
//   - each element must contain only lower-case alphanumeric characters or dashes
//
// All errors returned by this function will be "invalid" type errors. If the
// caller wants better errors, it must take responsibility for checking things
// like required/optional and max-length.
func LongName[T ~string](_ context.Context, op operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value == nil {
		return nil
	}
	var allErrs field.ErrorList
	for _, msg := range content.IsDNS1123Subdomain((string)(*value)) {
		allErrs = append(allErrs, field.Invalid(fldPath, *value, msg).WithOrigin("format=k8s-long-name"))
	}
	return allErrs
}

// UUID verifies that the specified value is a valid UUID (RFC 4122).
//   - must be 36 characters long
//   - must be in the normalized form `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
//   - must use only lowercase hexadecimal characters
func UUID[T ~string](_ context.Context, op operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value == nil {
		return nil
	}
	val := (string)(*value)
	if len(val) != 36 {
		return field.ErrorList{field.Invalid(fldPath, val, uuidErrorMessage).WithOrigin("format=k8s-uuid")}
	}
	for idx := 0; idx < len(val); idx++ {
		character := val[idx]
		switch idx {
		case 8, 13, 18, 23:
			if character != '-' {
				return field.ErrorList{field.Invalid(fldPath, val, uuidErrorMessage).WithOrigin("format=k8s-uuid")}
			}
		default:
			// should be lower case hexadecimal.
			if (character < '0' || character > '9') && (character < 'a' || character > 'f') {
				return field.ErrorList{field.Invalid(fldPath, val, uuidErrorMessage).WithOrigin("format=k8s-uuid")}
			}
		}
	}
	return nil
}
