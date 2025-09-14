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

	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// NoSetValue verifies that a field cannot be set (transition from unset to set) for comparable types.
func NoSetValue[T comparable](_ context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *T) field.ErrorList {
	if op.Type != operation.Update {
		return nil
	}

	var zero T
	if *oldValue == zero && *value != zero {
		return field.ErrorList{
			field.Forbidden(fldPath, "field cannot be set once created"),
		}
	}

	return nil
}

// NoSetPointer verifies that a pointer field cannot be set (transition from nil to non-nil).
func NoSetPointer[T any](_ context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *T) field.ErrorList {
	if op.Type != operation.Update {
		return nil
	}

	if oldValue == nil && value != nil {
		return field.ErrorList{
			field.Forbidden(fldPath, "field cannot be set once created"),
		}
	}

	return nil
}

// NoUnsetValue verifies that a field cannot be unset (transition from set to unset) for comparable types.
func NoUnsetValue[T comparable](_ context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *T) field.ErrorList {
	if op.Type != operation.Update {
		return nil
	}

	var zero T
	if *oldValue != zero && *value == zero {
		return field.ErrorList{
			field.Forbidden(fldPath, "field cannot be cleared once set"),
		}
	}

	return nil
}

// NoUnsetPointer verifies that a pointer field cannot be unset (transition from non-nil to nil).
func NoUnsetPointer[T any](_ context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *T) field.ErrorList {
	if op.Type != operation.Update {
		return nil
	}

	if oldValue != nil && value == nil {
		return field.ErrorList{
			field.Forbidden(fldPath, "field cannot be cleared once set"),
		}
	}

	return nil
}

// NoModifyValue verifies that a field's value cannot be modified (but allows set/unset transitions) for comparable types.
// This uses direct comparison for performance, similar to ImmutableByCompare.
func NoModifyValue[T comparable](_ context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *T) field.ErrorList {
	if op.Type != operation.Update {
		return nil
	}

	var zero T
	// Allow transitions between set/unset
	if *oldValue == zero || *value == zero {
		return nil
	}

	// Both are set - use direct comparison for performance
	if *value != *oldValue {
		return field.ErrorList{
			field.Forbidden(fldPath, "field cannot be modified once set"),
		}
	}

	return nil
}

// NoModifyValueByReflect verifies that a field's value cannot be modified (but allows set/unset transitions).
// This uses DeepEqual for types that are not directly comparable.
func NoModifyValueByReflect[T any](_ context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *T) field.ErrorList {
	if op.Type != operation.Update {
		return nil
	}

	// Check if values are zero using reflection
	var zero T
	valueIsZero := equality.Semantic.DeepEqual(*value, zero)
	oldValueIsZero := equality.Semantic.DeepEqual(*oldValue, zero)

	// Allow transitions between set/unset
	if oldValueIsZero || valueIsZero {
		return nil
	}

	// Both are set - check if they're equal using DeepEqual
	if !equality.Semantic.DeepEqual(*value, *oldValue) {
		return field.ErrorList{
			field.Forbidden(fldPath, "field cannot be modified once set"),
		}
	}

	return nil
}

// NoModifyPointer verifies that a pointer field's value cannot be modified (but allows set/unset transitions).
func NoModifyPointer[T any](_ context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *T) field.ErrorList {
	if op.Type != operation.Update {
		return nil
	}

	// Allow transitions between set/unset
	if oldValue == nil || value == nil {
		return nil
	}

	// Both are set - check if they're equal using DeepEqual for the pointed values
	if !equality.Semantic.DeepEqual(value, oldValue) {
		return field.ErrorList{
			field.Forbidden(fldPath, "field cannot be modified once set"),
		}
	}

	return nil
}

// NoModifyStruct verifies that a non-pointer struct field cannot be modified.
// Non-pointer structs are always considered "set" and never "unset".
func NoModifyStruct[T any](_ context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *T) field.ErrorList {
	if op.Type != operation.Update {
		return nil
	}

	// For structs, we always check equality since they can't be unset
	if !equality.Semantic.DeepEqual(value, oldValue) {
		return field.ErrorList{
			field.Forbidden(fldPath, "field cannot be modified once set"),
		}
	}

	return nil
}
