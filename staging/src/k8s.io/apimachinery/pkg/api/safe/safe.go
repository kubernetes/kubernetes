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

// Field takes a pointer to any value (which may or may not be nil) and a
// function that traverses to a target type R (a typical use case is to
// dereference a field), and returns the result of the traversal, or the zero
// value of the target type.
//
// This is roughly equivalent to:
//
//	value != nil ? fn(value) : zero-value
//
// ...in languages that support the ternary operator.
func Field[V any, R any](value *V, fn func(*V) R) R {
	if value == nil {
		var zero R
		return zero
	}
	o := fn(value)
	return o
}

// Cast takes any value, attempts to cast it to T, and returns the T value if
// the cast is successful, or else the zero value of T.
func Cast[T any](value any) T {
	result, _ := value.(T)
	return result
}

// Value takes a pointer to any value (which may or may not be nil) and a
// function that returns a pointer to the same type.  If the value is not nil,
// it is returned, otherwise the result of the function is returned.
//
// This is roughly equivalent to:
//
//	value != nil ? value : fn()
//
// ...in languages that support the ternary operator.
func Value[T any](value *T, fn func() *T) *T {
	if value != nil {
		return value
	}
	return fn()
}
