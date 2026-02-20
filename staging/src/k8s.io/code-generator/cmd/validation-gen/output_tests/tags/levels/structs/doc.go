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

package structs

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type MixedStruct struct {
	TypeMeta int

	// +k8s:minimum=5
	// +k8s:alpha=+k8s:minimum=10
	IntField int `json:"intField"`

	// +k8s:minimum=5
	// +k8s:beta=+k8s:minimum=10
	IntFieldBeta int `json:"intFieldBeta"`

	// +k8s:maxItems=5
	// +k8s:alpha=+k8s:maxItems=3
	ListField []string `json:"listField"`

	// +k8s:maxItems=5
	// +k8s:beta=+k8s:maxItems=3
	ListFieldBeta []string `json:"listFieldBeta"`
}

type ConditionalStruct struct {
	TypeMeta int

	// +k8s:alpha=+k8s:ifEnabled(MyFeature)=+k8s:minimum=10
	ConditionalField int `json:"conditionalField"`

	// +k8s:beta=+k8s:ifEnabled(MyFeature)=+k8s:minimum=10
	ConditionalFieldBeta int `json:"conditionalFieldBeta"`

	// +k8s:alpha=+k8s:alpha=+k8s:minimum=20
	RecursiveAlpha int `json:"recursiveAlpha"`

	// +k8s:beta=+k8s:beta=+k8s:minimum=20
	RecursiveBeta int `json:"recursiveBeta"`
}
