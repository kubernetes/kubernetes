// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package conv

// Unlicensed credits (idea, concept)
//
// The idea to convert values to pointers and the other way around, was inspired, eons ago, by the aws go sdk.
//
// Nowadays, all sensible API sdk's expose a similar functionality.

// Pointer returns a pointer to the value passed in.
func Pointer[T any](v T) *T {
	return &v
}

// Value returns a shallow copy of the value of the pointer passed in.
//
// If the pointer is nil, the returned value is the zero value.
func Value[T any](v *T) T {
	if v != nil {
		return *v
	}

	var zero T
	return zero
}

// PointerSlice converts a slice of values into a slice of pointers.
func PointerSlice[T any](src []T) []*T {
	dst := make([]*T, len(src))
	for i := 0; i < len(src); i++ {
		dst[i] = &(src[i])
	}
	return dst
}

// ValueSlice converts a slice of pointers into a slice of values.
//
// nil elements are zero values.
func ValueSlice[T any](src []*T) []T {
	dst := make([]T, len(src))
	for i := 0; i < len(src); i++ {
		if src[i] != nil {
			dst[i] = *(src[i])
		}
	}
	return dst
}

// PointerMap converts a map of values into a map of pointers.
func PointerMap[K comparable, T any](src map[K]T) map[K]*T {
	dst := make(map[K]*T)
	for k, val := range src {
		v := val
		dst[k] = &v
	}
	return dst
}

// ValueMap converts a map of pointers into a map of values.
//
// nil elements are skipped.
func ValueMap[K comparable, T any](src map[K]*T) map[K]T {
	dst := make(map[K]T)
	for k, val := range src {
		if val != nil {
			dst[k] = *val
		}
	}
	return dst
}
