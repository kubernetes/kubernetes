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
	"k8s.io/apimachinery/pkg/api/validate/constraints"
	"k8s.io/apimachinery/pkg/api/validate/content"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// MaxLength verifies that the specified value is not longer than max
// characters.
func MaxLength[T ~string](_ context.Context, _ operation.Operation, fldPath *field.Path, value, _ *T, max int) field.ErrorList {
	if value == nil {
		return nil
	}
	if len(*value) > max {
		return field.ErrorList{field.TooLong(fldPath, *value, max).WithOrigin("maxLength")}
	}
	return nil
}

// MaxItems verifies that the specified slice is not longer than max items.
func MaxItems[T any](_ context.Context, _ operation.Operation, fldPath *field.Path, value, _ []T, max int) field.ErrorList {
	if len(value) > max {
		return field.ErrorList{field.TooMany(fldPath, len(value), max).WithOrigin("maxItems")}
	}
	return nil
}

// Minimum verifies that the specified value is greater than or equal to min.
func Minimum[T constraints.Integer](_ context.Context, _ operation.Operation, fldPath *field.Path, value, _ *T, min T) field.ErrorList {
	if value == nil {
		return nil
	}
	if *value < min {
		return field.ErrorList{field.Invalid(fldPath, *value, content.MinError(min)).WithOrigin("minimum")}
	}
	return nil
}
