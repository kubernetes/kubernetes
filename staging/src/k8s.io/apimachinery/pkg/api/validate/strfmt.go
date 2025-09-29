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
	"fmt"
	"strings"

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

// LabelKey verifies that the specified value is a valid label key.
// A label key is composed of an optional prefix and a name, separated by a '/'.
// The name part is required and must:
//   - be 63 characters or less
//   - begin and end with an alphanumeric character ([a-z0-9A-Z])
//   - contain only alphanumeric characters, dashes (-), underscores (_), or dots (.)
//
// The prefix is optional and must:
//   - be a DNS subdomain
//   - be no more than 253 characters
func LabelKey[T ~string](_ context.Context, op operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value == nil {
		return nil
	}
	var allErrs field.ErrorList
	for _, msg := range content.IsLabelKey((string)(*value)) {
		allErrs = append(allErrs, field.Invalid(fldPath, *value, msg).WithOrigin("format=k8s-label-key"))
	}
	return allErrs
}

// LongNameCaseless verifies that the specified value is a valid "long name"
// (sometimes known as a "DNS subdomain"), but is case-insensitive.
//   - must not be empty
//   - must be less than 254 characters long
//   - each element must start and end with alphanumeric characters
//   - each element must contain only alphanumeric characters or dashes
//
// Deprecated: Case-insensitive names are not recommended as they can lead to ambiguity
// (e.g., 'Foo', 'FOO', and 'foo' would be allowed names for foo). Use LongName for strict, lowercase validation.
func LongNameCaseless[T ~string](_ context.Context, op operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value == nil {
		return nil
	}
	var allErrs field.ErrorList
	for _, msg := range content.IsDNS1123SubdomainCaseless((string)(*value)) {
		allErrs = append(allErrs, field.Invalid(fldPath, *value, msg).WithOrigin("format=k8s-long-name-caseless"))
	}
	return allErrs
}

// LabelValue verifies that the specified value is a valid label value.
//   - can be empty
//   - must be no more than 63 characters
//   - must start and end with alphanumeric characters
//   - must contain only alphanumeric characters, dashes, underscores, or dots
func LabelValue[T ~string](_ context.Context, op operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value == nil {
		return nil
	}
	var allErrs field.ErrorList
	for _, msg := range content.IsLabelValue((string)(*value)) {
		allErrs = append(allErrs, field.Invalid(fldPath, *value, msg).WithOrigin("format=k8s-label-value"))
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

// ResourcePoolName verifies that the specified value is one or more valid "long name"
// parts separated by a '/' and no longer than 253 characters.
func ResourcePoolName[T ~string](ctx context.Context, op operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value == nil {
		return nil
	}
	val := (string)(*value)
	var allErrs field.ErrorList
	if len(val) > 253 {
		allErrs = append(allErrs, field.TooLong(fldPath, val, 253))
	}
	parts := strings.Split(val, "/")
	for i, part := range parts {
		if len(part) == 0 {
			allErrs = append(allErrs, field.Invalid(fldPath, val, fmt.Sprintf("segment %d: must not be empty", i)))
			continue
		}
		// Note that we are overwriting the origin from the underlying LongName validation.
		allErrs = append(allErrs, LongName(ctx, op, fldPath, &part, nil).PrefixDetail(fmt.Sprintf("segment %d: ", i))...)
	}
	return allErrs.WithOrigin("format=k8s-resource-pool-name")
}
