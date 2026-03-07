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
	"math"
	"unicode/utf8"

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

	// if the length of the value in bytes is less
	// than the maximum size then we can confidently
	// say that this value is within the bounds
	// enforced by the maximum value regardless
	// of the actual makeup of characters in the value
	byteLength := len(*value)
	if byteLength <= max {
		return nil
	}

	// because runes are up to 4 byte characters, if we assume all characters
	// in the input are runes, the minimum number of characters that
	// are specified is len(value)/4. If the minimum multi-byte
	// character count is greater than our enforced maximum, we
	// can confidently say that the value is invalid without having
	// to actually perform the more expensive rune counting step
	minimum := int(math.Ceil(float64(byteLength) / 4.0))
	if minimum > max || utf8.RuneCountInString(string(*value)) > max {
		return field.ErrorList{field.TooLongCharacters(fldPath, *value, max).WithOrigin("maxLength")}
	}
	return nil
}

// MaxBytes verifies that the specified value is not longer than max bytes.
func MaxBytes[T ~string](_ context.Context, _ operation.Operation, fldPath *field.Path, value, _ *T, max int) field.ErrorList {
	if value == nil {
		return nil
	}

	if len(*value) > max {
		return field.ErrorList{field.TooLong(fldPath, *value, max).WithOrigin("maxBytes")}
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

// MinLength verifies that the specified value is at least min characters, if non-nil.
func MinLength[T ~string](_ context.Context, _ operation.Operation, fldPath *field.Path, value, _ *T, min int) field.ErrorList {
	if value == nil {
		return nil
	}

	byteLength := len(*value)

	// because runes are up to 4 byte characters, if we assume all characters
	// in the input are 4 byte runes, the minimum number of characters that
	// are specified is len(value)/4. If the minimum multi-byte
	// character count is greater than or equal to our enforced minimum, we
	// can confidently say that the value is valid without having
	// to actually perform the more expensive rune counting step
	if int(math.Ceil(float64(byteLength)/4.0)) >= min {
		return nil
	}

	// if the length of the value in bytes is less
	// than the minimum size then we can confidently
	// say that this value is not within the bounds
	// enforced by the maximum value regardless
	// of the actual makeup of characters in the value.
	// Otherwise, perform a rune count to determine if the
	// number of characters is less than the minimum.
	if byteLength < min || utf8.RuneCountInString(string(*value)) < min {
		return field.ErrorList{field.TooShort(fldPath, *value, min).WithOrigin("minLength")}
	}
	return nil
}
