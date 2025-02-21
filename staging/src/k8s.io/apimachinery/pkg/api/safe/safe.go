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

package safe

// Field takes a pointer to any value (which may or may not be nil) and
// a function that traverses to a target type R (a typical use case is to dereference a field),
// and returns the result of the traversal, or the zero value of the target type.
// This is roughly equivalent to "value != nil ? fn(value) : nil" in languages that support the ternary operator.
func Field[V any, R any](value *V, fn func(*V) R) R {
	if value == nil {
		var zero R
		return zero
	}
	o := fn(value)
	return o
}

// Lookup looks up the specified key in the map and, if found, applies a
// transform function.  If the map is nil or the key is not found, it returns
// the zero-value for the map's value type.  The transform function will only
// be called if the map is non-nil and the key is found.
func Lookup[K comparable, V any, R any](m map[K]V, key K, xform func(V) R) R {
	var zero R
	if m == nil {
		return zero
	}
	val, found := m[key]
	if !found {
		return zero
	}
	return xform(val)
}

// PtrTo returns a pointer to the argument value.  This works as a transform
// function for Lookup, e.g. given a map with non-pointer values, this will
// produce a pointer.
func PtrTo[T any](val T) *T {
	return &val
}

// Deref returns the value pointed to by the argument pointer, or the zero
// value if the pointer is nil.  This works as a transform function for Lookup,
// e.g. given a map with pointer-to-slice values, this will produce a slice.
func Deref[T any](val *T) T {
	if val == nil {
		var zero T
		return zero
	}
	return *val
}

// Ident returns the input argument.  This works as a transform function for
// Lookup, e.g. given a map with pointer values, this will return the pointer
// directly.
func Ident[T any](val T) T {
	return val
}

// Cast takes any value, attempts to cast it to T, and returns the T value if
// the cast is successful, or else the zero value of T.
func Cast[T any](value any) T {
	result, _ := value.(T)
	return result
}
