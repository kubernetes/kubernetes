//go:build go1.18
// +build go1.18

/*
Copyright 2023 The Kubernetes Authors.

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

package klog

// SafePtr is a function that takes a pointer of any type (T) as an argument.
// If the provided pointer is not nil, it returns the same pointer. If it is nil, it returns nil instead.
//
// This function is particularly useful to prevent nil pointer dereferencing when:
//
//   - The type implements interfaces that are called by the logger, such as `fmt.Stringer`.
//   - And these interface implementations do not perform nil checks themselves.
func SafePtr[T any](p *T) any {
	if p == nil {
		return nil
	}
	return p
}
