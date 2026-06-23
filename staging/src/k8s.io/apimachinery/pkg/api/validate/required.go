/*
Copyright 2024 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// RequiredValue verifies that the specified value is not the zero-value for
// its type.
func RequiredValue[T comparable](_ context.Context, _ operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	var zero T
	if *value != zero {
		return nil
	}
	return field.ErrorList{field.Required(fldPath, "")}
}

// RequiredPointer verifies that the specified pointer is not nil.
func RequiredPointer[T any](_ context.Context, _ operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value != nil {
		return nil
	}
	return field.ErrorList{field.Required(fldPath, "")}
}

// RequiredSlice verifies that the specified slice is not empty.
func RequiredSlice[T any](_ context.Context, _ operation.Operation, fldPath *field.Path, value, _ []T) field.ErrorList {
	if len(value) > 0 {
		return nil
	}
	return field.ErrorList{field.Required(fldPath, "")}
}

// RequiredMap verifies that the specified map is not empty.
func RequiredMap[K comparable, T any](_ context.Context, _ operation.Operation, fldPath *field.Path, value, _ map[K]T) field.ErrorList {
	if len(value) > 0 {
		return nil
	}
	return field.ErrorList{field.Required(fldPath, "")}
}

// ForbiddenValue verifies that the specified value is the zero-value for its
// type.
func ForbiddenValue[T comparable](_ context.Context, _ operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	var zero T
	if *value == zero {
		return nil
	}
	return field.ErrorList{field.Forbidden(fldPath, "")}
}

// ForbiddenPointer verifies that the specified pointer is nil.
func ForbiddenPointer[T any](_ context.Context, _ operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value == nil {
		return nil
	}
	return field.ErrorList{field.Forbidden(fldPath, "")}
}

// ForbiddenSlice verifies that the specified slice is empty.
func ForbiddenSlice[T any](_ context.Context, _ operation.Operation, fldPath *field.Path, value, _ []T) field.ErrorList {
	if len(value) == 0 {
		return nil
	}
	return field.ErrorList{field.Forbidden(fldPath, "")}
}

// ForbiddenMap verifies that the specified map is empty.
func ForbiddenMap[K comparable, T any](_ context.Context, _ operation.Operation, fldPath *field.Path, value, _ map[K]T) field.ErrorList {
	if len(value) == 0 {
		return nil
	}
	return field.ErrorList{field.Forbidden(fldPath, "")}
}

// OptionalValue reports whether the specified value is non-zero (i.e. present).
// Returns true if the value is set, false if it is the zero value for its type.
// The generated code uses this to skip further validation when an optional field
// is absent; a false return is not an error.
func OptionalValue[T comparable](_ context.Context, _ operation.Operation, _ *field.Path, value, _ *T) bool {
	var zero T
	return *value != zero
}

// OptionalPointer reports whether the specified pointer is non-nil (i.e. present).
// Returns true if the pointer is set, false if it is nil.
// The generated code uses this to skip further validation when an optional field
// is absent; a false return is not an error.
func OptionalPointer[T any](_ context.Context, _ operation.Operation, _ *field.Path, value, _ *T) bool {
	return value != nil
}

// OptionalSlice reports whether the specified slice is non-empty (i.e. present).
// Returns true if the slice has at least one element, false if it is nil or empty.
// The generated code uses this to skip further validation when an optional field
// is absent; a false return is not an error.
func OptionalSlice[T any](_ context.Context, _ operation.Operation, _ *field.Path, value, _ []T) bool {
	return len(value) > 0
}

// OptionalMap reports whether the specified map is non-empty (i.e. present).
// Returns true if the map has at least one entry, false if it is nil or empty.
// The generated code uses this to skip further validation when an optional field
// is absent; a false return is not an error.
func OptionalMap[K comparable, T any](_ context.Context, _ operation.Operation, _ *field.Path, value, _ map[K]T) bool {
	return len(value) > 0
}
