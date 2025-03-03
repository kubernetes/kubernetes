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

// +k8s:validation-gen=TypeMeta
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme
// +k8s:validation-gen-test-fixture=validateFalse

// This is a test package.
package pointers

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type T1 struct {
	TypeMeta int

	// +k8s:validateFalse="field T1.PS"
	PS *string `json:"ps"`
	// +k8s:validateFalse="field T1.PI"
	PI *int `json:"pi"`
	// +k8s:validateFalse="field T1.PB"
	PB *bool `json:"pb"`
	// +k8s:validateFalse="field T1.PF"
	PF *float64 `json:"pf"`

	// +k8s:validateFalse="field T1.PT2"
	PT2 *T2 `json:"pt2"`

	// Duplicate types with no validation.
	AnotherPS *string  `json:"anotherps"`
	AnotherPI *int     `json:"anotherpi"`
	AnotherPB *bool    `json:"anotherpb"`
	AnotherPF *float64 `json:"anotherpf"`
}

// Note: This has validations and is linked into the type-graph of T1.
type T2 struct {
	// +k8s:validateFalse="field T2.PS"
	PS *string `json:"ps"`
	// +k8s:validateFalse="field T2.PI"
	PI *int `json:"pi"`
	// +k8s:validateFalse="field T2.PB"
	PB *bool `json:"pb"`
	// +k8s:validateFalse="field T2.PF"
	PF *float64 `json:"pf"`
}
