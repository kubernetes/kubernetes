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
	"sort"

	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// MatchFunc is a function that compares two values of the same type,
// according to some criteria, and returns true if they match.
type MatchFunc[T any] func(T, T) bool

// EachSliceVal performs validation on each element of newSlice using the provided validation function.
//
// For update operations, the match function finds corresponding values in oldSlice for each
// value in newSlice. This comparison can be either full or partial (e.g., matching only
// specific struct fields that serve as a unique identifier). If match is nil, validation
// proceeds without considering old values, and the equiv function is not used.
//
// For update operations, the equiv function checks if a new value is equivalent to its
// corresponding old value, enabling validation ratcheting. If equiv is nil but match is
// provided, the match function is assumed to perform full value comparison.
//
// Note: The slice element type must be non-nilable.
func EachSliceVal[T any](ctx context.Context, op operation.Operation, fldPath *field.Path, newSlice, oldSlice []T,
	match, equiv MatchFunc[T], validator ValidateFunc[*T]) field.ErrorList {
	var errs field.ErrorList
	for i, val := range newSlice {
		var old *T
		if match != nil && len(oldSlice) > 0 {
			old = lookup(oldSlice, val, match)
		}
		// If the operation is an update, for validation ratcheting, skip re-validating if the old
		// value exists and either:
		// 1. The match function provides full comparison (equiv is nil)
		// 2. The equiv function confirms the values are equivalent (either directly or semantically)
		//
		// The equiv function provides equality comparison when match uses partial comparison.
		if op.Type == operation.Update && old != nil && (equiv == nil || equiv(val, *old)) {
			continue
		}
		errs = append(errs, validator(ctx, op, fldPath.Index(i), &val, old)...)
	}
	return errs
}

// lookup returns a pointer to the first element in the list that matches the
// target, according to the provided comparison function, or else nil.
func lookup[T any](list []T, target T, match MatchFunc[T]) *T {
	for i := range list {
		if match(list[i], target) {
			return &list[i]
		}
	}
	return nil
}

// EachMapVal validates each value in newMap using the specified validation
// function, passing the corresponding old value from oldMap if the key exists in oldMap.
// For update operations, it implements validation ratcheting by skipping validation
// when the old value exists and the equiv function confirms the values are equivalent.
// The value-type of the map is assumed to not be nilable.
// If equiv is nil, value-based ratcheting is disabled and all values will be validated.
func EachMapVal[K ~string, V any](ctx context.Context, op operation.Operation, fldPath *field.Path, newMap, oldMap map[K]V,
	equiv MatchFunc[V], validator ValidateFunc[*V]) field.ErrorList {
	var errs field.ErrorList
	for key, val := range newMap {
		var old *V
		if o, found := oldMap[key]; found {
			old = &o
		}
		// If the operation is an update, for validation ratcheting, skip re-validating if the old
		// value is found and the equiv function confirms the values are equivalent.
		if op.Type == operation.Update && old != nil && equiv != nil && equiv(val, *old) {
			continue
		}
		errs = append(errs, validator(ctx, op, fldPath.Key(string(key)), &val, old)...)
	}
	return errs
}

// EachMapKey validates each element of newMap with the specified
// validation function.
func EachMapKey[K ~string, T any](ctx context.Context, op operation.Operation, fldPath *field.Path, newMap, oldMap map[K]T,
	validator ValidateFunc[*K]) field.ErrorList {
	var errs field.ErrorList
	for key := range newMap {
		var old *K
		if _, found := oldMap[key]; found {
			old = &key
		}
		// If the operation is an update, for validation ratcheting, skip re-validating if
		// the key is found in oldMap.
		if op.Type == operation.Update && old != nil {
			continue
		}
		// Note: the field path is the field, not the key.
		errs = append(errs, validator(ctx, op, fldPath, &key, nil)...)
	}
	return errs
}

// Unique verifies that each element of newSlice is unique, according to the
// match function. It compares every element of the slice with every other
// element and returns errors for non-unique items.
func Unique[T any](_ context.Context, _ operation.Operation, fldPath *field.Path, newSlice, _ []T, match MatchFunc[T]) field.ErrorList {
	var dups []int
	for i, val := range newSlice {
		for j := i + 1; j < len(newSlice); j++ {
			other := newSlice[j]
			if match(val, other) {
				if dups == nil {
					dups = make([]int, 0, len(newSlice))
				}
				if lookup(dups, j, func(a, b int) bool { return a == b }) == nil {
					dups = append(dups, j)
				}
			}
		}
	}

	var errs field.ErrorList
	sort.Ints(dups)
	for _, i := range dups {
		var val any = newSlice[i]
		// TODO: we don't want the whole item to be logged in the error, just
		// the key(s). Unfortunately, the way errors are rendered, it comes out
		// as something like "map[string]any{...}" which is not very nice. Once
		// that is fixed, we can consider adding a way for this function to
		// specify that just the keys should be rendered in the error.
		errs = append(errs, field.Duplicate(fldPath.Index(i), val))
	}
	return errs
}

// SemanticDeepEqual is a MatchFunc that uses equality.Semantic.DeepEqual to
// compare two values.
// This wrapper is needed because MatchFunc requires a function that takes two
// arguments of specific type T, while equality.Semantic.DeepEqual takes
// arguments of type interface{}/any. The wrapper satisfies the type
// constraints of MatchFunc while leveraging the underlying semantic equality
// logic. It can be used by any other function that needs to call DeepEqual.
func SemanticDeepEqual[T any](a, b T) bool {
	return equality.Semantic.DeepEqual(a, b)
}

// DirectEqual is a MatchFunc that uses the == operator to compare two values.
// It can be used by any other function that needs to compare two values
// directly.
func DirectEqual[T comparable](a, b T) bool {
	return a == b
}
