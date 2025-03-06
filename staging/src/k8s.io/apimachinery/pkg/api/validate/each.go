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

// CompareFunc is a function that compares two values of the same type.
type CompareFunc[T any] func(T, T) bool

// EachSliceVal validates each element of newSlice with the specified
// validation function.  The comparison function is used to find the
// corresponding value in oldSlice.  The value-type of the slices is assumed to
// not be nilable.
func EachSliceVal[T any](ctx context.Context, op operation.Operation, fldPath *field.Path, newSlice, oldSlice []T,
	cmp CompareFunc[T], validator ValidateFunc[*T]) field.ErrorList {
	var errs field.ErrorList
	for i, val := range newSlice {
		var old *T
		if cmp != nil && len(oldSlice) > 0 {
			old = lookup(oldSlice, val, cmp)
		}
		errs = append(errs, validator(ctx, op, fldPath.Index(i), &val, old)...)
	}
	return errs
}

// lookup returns a pointer to the first element in the list that matches the
// target, according to the provided comparison function, or else nil.
func lookup[T any](list []T, target T, cmp func(T, T) bool) *T {
	for i := range list {
		if cmp(list[i], target) {
			return &list[i]
		}
	}
	return nil
}

// EachMapVal validates each element of newMap with the specified validation
// function and, if the corresponding key is found in oldMap, the old value.
// The value-type of the slices is assumed to not be nilable.
func EachMapVal[K ~string, V any](ctx context.Context, op operation.Operation, fldPath *field.Path, newMap, oldMap map[K]V,
	validator ValidateFunc[*V]) field.ErrorList {
	var errs field.ErrorList
	for key, val := range newMap {
		var old *V
		if o, found := oldMap[key]; found {
			old = &o
		}
		errs = append(errs, validator(ctx, op, fldPath.Key(string(key)), &val, old)...)
	}
	return errs
}

// EachMapKey validates each element of newMap with the specified
// validation function.  The oldMap argument is not used.
func EachMapKey[K ~string, T any](ctx context.Context, op operation.Operation, fldPath *field.Path, newMap, oldMap map[K]T,
	validator ValidateFunc[*K]) field.ErrorList {
	var errs field.ErrorList
	for key := range newMap {
		// Note: the field path is the field, not the key.
		errs = append(errs, validator(ctx, op, fldPath, &key, nil)...)
	}
	return errs
}
