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

// This is a test package.
package recursive

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

// +k8s:validateFalse="type T1"
type T1 struct {
	// +k8s:validateFalse="field T1.PT1"
	// +k8s:optional
	PT1 *T1 `json:"pt1"`

	// +k8s:validateFalse="field T1.T2"
	T2 T2 `json:"t2"`
	// +k8s:validateFalse="field T1.PT2"
	// +k8s:optional
	PT2 *T2 `json:"pt2"`
}

// +k8s:validateFalse="type T2"
type T2 struct {
	// +k8s:validateFalse="field T2.PT1"
	// +k8s:optional
	PT1 *T1 `json:"pt1"`

	// +k8s:validateFalse="field T2.PT2"
	// +k8s:optional
	PT2 *T2 `json:"pt2"`
}

// +k8s:validateFalse="type E1"
// +k8s:eachVal=+k8s:validateFalse="type E1 values"
type E1 []E1

// NOTE: no validations.
type T3 struct {
	// NOTE: no validations.
	PT3 *T3 `json:"pt3"`
}
