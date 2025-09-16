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
package slices

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

// This test case is carefully constructed to test recursion. We don't want to
// add more `validateFalse` tags because the bug that motivated this test
// wasn't looking deep enough into the recursion tree.
//
// Expectations:
// * We should emit validation for T1 because T3 has validation.
// * We should emit validation for T2 because it uses T1, which has validation.
// * We should emit validation for T3 because it has validation.
// * We should emit validation for T4 because it uses T3, which has validation.
// * T1 should call T2 and T3.
// * T2 should call eachVal(T1).
// * T3 should call T4.
// * T4 should call eachVal(T3).

type T1 struct {
	T2 T2 `json:"t2"`
	T3 T3 `json:"t3"`
}

type T2 struct {
	ST1 []T1 `json:"st1"`
}

// +k8s:validateFalse="type T3"
type T3 struct {
	T4 T4 `json:"t4"`
}

type T4 struct {
	ST3 []T3 `json:"st3"`
}
