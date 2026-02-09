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

package mapvalidation

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type MapValidationStruct struct {
	TypeMeta int

	// Case: Standard eachVal
	// +k8s:eachVal=+k8s:maxLength=2
	StandardEachVal map[string]string `json:"standardEachVal"`

	// Case: Alpha eachVal
	// +k8s:alpha=+k8s:eachVal=+k8s:maxLength=2
	AlphaEachVal map[string]string `json:"AlphaEachVal"`

	// Case: Beta eachVal
	// +k8s:beta=+k8s:eachVal=+k8s:maxLength=2
	BetaEachVal map[string]string `json:"BetaEachVal"`

	// Case: Standard eachKey
	// +k8s:eachKey=+k8s:maxLength=2
	StandardEachKey map[string]string `json:"standardEachKey"`

	// Case: Alpha eachKey
	// +k8s:alpha=+k8s:eachKey=+k8s:maxLength=2
	AlphaEachKey map[string]string `json:"AlphaEachKey"`

	// Case: Beta eachKey
	// +k8s:beta=+k8s:eachKey=+k8s:maxLength=2
	BetaEachKey map[string]string `json:"BetaEachKey"`

	// Case: Alpha Validation
	// +k8s:eachVal=+k8s:alpha=+k8s:maxLength=2
	AlphaValidation map[string]string `json:"AlphaValidation"`

	// Case: Beta Validation
	// +k8s:eachVal=+k8s:beta=+k8s:maxLength=2
	BetaValidation map[string]string `json:"BetaValidation"`
}
