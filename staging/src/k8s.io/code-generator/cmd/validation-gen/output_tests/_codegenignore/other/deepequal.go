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

package other

import "reflect"

// DeepEqualCalls counts calls to DeepEqual, so tests can assert it was used.
var DeepEqualCalls int

// DeepEqual lets a test package point +k8s:validation-gen-deep-equal-func at a
// function outside the package being generated into.
func DeepEqual[T any](a, b T) bool {
	DeepEqualCalls++
	return reflect.DeepEqual(a, b)
}
