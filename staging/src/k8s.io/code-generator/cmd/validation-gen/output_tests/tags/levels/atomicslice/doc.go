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

// +k8s:validation-gen=TypeMeta
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme

package atomicslice

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type AtomicSliceStruct struct {
	TypeMeta int

	// Case: Standard eachVal
	// +k8s:listType=atomic
	// +k8s:eachVal=+k8s:minimum=10
	Standard []int `json:"standard"`

	// Case: Alpha eachVal
	// +k8s:listType=atomic
	// +k8s:alpha=+k8s:eachVal=+k8s:minimum=10
	Alpha []int `json:"Alpha"`

	// Case: Beta eachVal
	// +k8s:listType=atomic
	// +k8s:beta=+k8s:eachVal=+k8s:minimum=10
	Beta []int `json:"Beta"`

	// Case: Standard eachVal, Alpha validation
	// +k8s:listType=atomic
	// +k8s:eachVal=+k8s:alpha=+k8s:minimum=10
	AlphaValidation []int `json:"AlphaValidation"`

	// Case: Standard eachVal, Beta validation
	// +k8s:listType=atomic
	// +k8s:eachVal=+k8s:beta=+k8s:minimum=10
	BetaValidation []int `json:"BetaValidation"`
}
