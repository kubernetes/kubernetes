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
	"slices"

	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// UpdateConstraint represents a constraint on update operations
type UpdateConstraint int

const (
	// NoSet prevents unset->set transitions
	NoSet UpdateConstraint = iota
	// NoUnset prevents set->unset transitions
	NoUnset
	// NoModify prevents value changes but allows set/unset transitions
	NoModify
	// NoAddItem prevents adding items to a slice or map
	NoAddItem
	// NoRemoveItem prevents removing items from a slice or map
	NoRemoveItem
)

// UpdateValueByCompare verifies update constraints for comparable value types.
func UpdateValueByCompare[T comparable](_ context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *T, constraints ...UpdateConstraint) field.ErrorList {
	// nil oldValue means no prior value to compare against (eg: a new item at +k8s:eachVal scope) -> no transition to check.
	if op.Type != operation.Update || oldValue == nil {
		return nil
	}

	var errs field.ErrorList
	var zero T

	for _, constraint := range constraints {
		switch constraint {
		case NoSet:
			if *oldValue == zero && *value != zero {
				errs = append(errs, field.Invalid(fldPath, nil, "field cannot be set once created").WithOrigin("update"))
			}
		case NoUnset:
			if *oldValue != zero && *value == zero {
				errs = append(errs, field.Invalid(fldPath, nil, "field cannot be cleared once set").WithOrigin("update"))
			}
		case NoModify:
			// Rely on validation ratcheting to detect that the value has changed.
			// This check only verifies that the field was set in both the old and
			// new objects, confirming it was a modification, not a set/unset.
			if *oldValue != zero && *value != zero {
				errs = append(errs, field.Invalid(fldPath, nil, "field cannot be modified once set").WithOrigin("update"))
			}
		}
	}

	return errs
}

// UpdatePointer verifies update constraints for pointer types.
func UpdatePointer[T any](_ context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *T, constraints ...UpdateConstraint) field.ErrorList {
	if op.Type != operation.Update {
		return nil
	}

	var errs field.ErrorList

	for _, constraint := range constraints {
		switch constraint {
		case NoSet:
			if oldValue == nil && value != nil {
				errs = append(errs, field.Invalid(fldPath, nil, "field cannot be set once created").WithOrigin("update"))
			}
		case NoUnset:
			if oldValue != nil && value == nil {
				errs = append(errs, field.Invalid(fldPath, nil, "field cannot be cleared once set").WithOrigin("update"))
			}
		case NoModify:
			// Rely on validation ratcheting to detect that the value has changed.
			// This check only verifies that the field was non-nil in both the old
			// and new objects, confirming it was a modification, not a set/unset.
			if oldValue != nil && value != nil {
				errs = append(errs, field.Invalid(fldPath, nil, "field cannot be modified once set").WithOrigin("update"))
			}
		}
	}

	return errs
}

// UpdateValueByReflect verifies update constraints for non-comparable value types using reflection.
func UpdateValueByReflect[T any](_ context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *T, constraints ...UpdateConstraint) field.ErrorList {
	// nil oldValue means no prior value to compare against (eg: a new item at +k8s:eachVal scope) -> no transition to check.
	if op.Type != operation.Update || oldValue == nil {
		return nil
	}

	var errs field.ErrorList
	var zero T
	valueIsZero := equality.Semantic.DeepEqual(*value, zero)
	oldValueIsZero := equality.Semantic.DeepEqual(*oldValue, zero)

	for _, constraint := range constraints {
		switch constraint {
		case NoSet:
			if oldValueIsZero && !valueIsZero {
				errs = append(errs, field.Invalid(fldPath, nil, "field cannot be set once created").WithOrigin("update"))
			}
		case NoUnset:
			if !oldValueIsZero && valueIsZero {
				errs = append(errs, field.Invalid(fldPath, nil, "field cannot be cleared once set").WithOrigin("update"))
			}
		case NoModify:
			// Rely on validation ratcheting to detect that the value has changed.
			// This check only verifies that the field was set in both the old and
			// new objects, confirming it was a modification, not a set/unset.
			if !oldValueIsZero && !valueIsZero {
				errs = append(errs, field.Invalid(fldPath, nil, "field cannot be modified once set").WithOrigin("update"))
			}
		}
	}

	return errs
}

// UpdateStruct verifies update constraints for non-pointer struct types.
// Non-pointer structs are always considered "set" and never "unset".
func UpdateStruct[T any](_ context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *T, constraints ...UpdateConstraint) field.ErrorList {
	// nil oldValue means no prior value to compare against (eg: a new item at +k8s:eachVal scope) -> no transition to check.
	if op.Type != operation.Update || oldValue == nil {
		return nil
	}

	var errs field.ErrorList

	for _, constraint := range constraints {
		switch constraint {
		case NoSet, NoUnset:
			// These constraints don't apply to non-pointer structs
			// as they can't be unset. This should be caught at generation time.
			continue
		case NoModify:
			// Non-pointer structs are always considered "set". Therefore, any
			// change detected by validation ratcheting is a modification.
			// The deep equality check is redundant and has been removed.
			errs = append(errs, field.Invalid(fldPath, nil, "field cannot be modified once set").WithOrigin("update"))
		}
	}

	return errs
}

// UpdateSlice verifies update constraints for slice types.
// NoAddItem and NoRemoveItem use the match function to find corresponding
// elements between value and oldValue. NoSet and NoUnset treat len == 0 as
// "unset".
func UpdateSlice[T any](_ context.Context, op operation.Operation, fldPath *field.Path, value, oldValue []T, match MatchFunc[T], constraints ...UpdateConstraint) field.ErrorList {
	if op.Type != operation.Update {
		return nil
	}

	if match == nil && (slices.Contains(constraints, NoAddItem) || slices.Contains(constraints, NoRemoveItem)) {
		return field.ErrorList{field.InternalError(fldPath, fmt.Errorf("UpdateSlice: NoAddItem/NoRemoveItem require a non-nil match function"))}
	}

	var errs field.ErrorList

	for _, constraint := range constraints {
		switch constraint {
		case NoSet:
			if len(oldValue) == 0 && len(value) > 0 {
				errs = append(errs, field.Invalid(fldPath, nil, "field cannot be set once created").WithOrigin("update"))
			}
		case NoUnset:
			if len(oldValue) > 0 && len(value) == 0 {
				errs = append(errs, field.Invalid(fldPath, nil, "field cannot be cleared once set").WithOrigin("update"))
			}
		case NoAddItem:
			for i, newItem := range value {
				if lookup(oldValue, newItem, match) == nil {
					errs = append(errs, field.Forbidden(fldPath.Index(i), "item may not be added").WithOrigin("update"))
				}
			}
		case NoRemoveItem:
			for _, oldItem := range oldValue {
				if lookup(value, oldItem, match) == nil {
					errs = append(errs, field.Forbidden(fldPath, "item may not be removed").WithOrigin("update"))
				}
			}
		}
	}

	return errs
}

// UpdateMap verifies update constraints for map types.
// NoAddItem and NoRemoveItem compare keys between value and oldValue. NoSet
// and NoUnset treat len == 0 as "unset".
func UpdateMap[K comparable, V any](_ context.Context, op operation.Operation, fldPath *field.Path, value, oldValue map[K]V, constraints ...UpdateConstraint) field.ErrorList {
	if op.Type != operation.Update {
		return nil
	}

	var errs field.ErrorList

	for _, constraint := range constraints {
		switch constraint {
		case NoSet:
			if len(oldValue) == 0 && len(value) > 0 {
				errs = append(errs, field.Invalid(fldPath, nil, "field cannot be set once created").WithOrigin("update"))
			}
		case NoUnset:
			if len(oldValue) > 0 && len(value) == 0 {
				errs = append(errs, field.Invalid(fldPath, nil, "field cannot be cleared once set").WithOrigin("update"))
			}
		case NoAddItem:
			for k := range value {
				if _, ok := oldValue[k]; !ok {
					errs = append(errs, field.Forbidden(fldPath.Key(fmt.Sprintf("%v", k)), "item may not be added").WithOrigin("update"))
				}
			}
		case NoRemoveItem:
			for k := range oldValue {
				if _, ok := value[k]; !ok {
					errs = append(errs, field.Forbidden(fldPath.Key(fmt.Sprintf("%v", k)), "item may not be removed").WithOrigin("update"))
				}
			}
		}
	}

	return errs
}
