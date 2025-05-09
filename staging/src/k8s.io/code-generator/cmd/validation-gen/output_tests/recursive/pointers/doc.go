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

// +k8s:validation-gen=*
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme
// +k8s:validation-gen-test-fixture=validateFalse

// This is a test package.
package pointers

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

// This test case is carefully constructed to test recursion. We don't want to
// add more `validateFalse` tags because we want to test the recursion.
//
// Expectations:
// * We should emit validation for T1 because it uses T2 which uses T3, which has validation.
// * We should emit validation for T2 because it uses T3, which has validation.
// * We should emit validation for T3 because it has validation.
// * We should NOT emit validation for T4.
// * T1 should call optional(T1), T2 and optional(T2).
// * T2 should call optional(T1), optional(T2), and optional(T3).

type T1 struct {
	// +k8s:optional
	PT1 *T1 `json:"pt1"`

	T2 T2 `json:"t2"`

	// +k8s:optional
	PT2 *T2 `json:"pt2"`
}

type T2 struct {
	// +k8s:optional
	PT1 *T1 `json:"pt1"`

	// +k8s:optional
	PT2 *T2 `json:"pt2"`

	// +k8s:optional
	PT3 *T3 `json:"pt3"`
}

// +k8s:validateFalse="type T3"
type T3 struct {
	I int `json:"i"`
}

// NOTE: no validations.
type T4 struct {
	// NOTE: no validations.
	PT4 *T4 `json:"pt4"`
}
