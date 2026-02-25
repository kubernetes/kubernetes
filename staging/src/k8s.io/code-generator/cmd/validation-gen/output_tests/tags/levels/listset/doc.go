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

package listset

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type ListSetStruct struct {
	TypeMeta int

	// Case: Alpha listType=set
	// +k8s:alpha=+k8s:listType=set
	Set []ComplexSetItem `json:"set"`

	// Case: Beta listType=set
	// +k8s:beta=+k8s:listType=set
	BetaSet []ComplexSetItem `json:"betaSet"`

	// Case: Chained subfield validation
	// +k8s:listType=set
	// +k8s:eachVal=+k8s:subfield(value)=+k8s:alpha=+k8s:minimum=10
	ChainedSubfieldSet []SimpleSetItem `json:"chainedSubfieldSet"`

	// Case: Alpha listType=set, Beta item validation
	// +k8s:alpha=+k8s:listType=set
	SetBetaItem []ComplexSetItemBeta `json:"setBetaItem"`

	// Case: Beta listType=set, Beta item validation
	// +k8s:beta=+k8s:listType=set
	BetaSetBetaItem []ComplexSetItemBeta `json:"betaSetBetaItem"`

	// Case: Chained subfield validation (Beta)
	// +k8s:listType=set
	// +k8s:eachVal=+k8s:subfield(value)=+k8s:beta=+k8s:minimum=10
	ChainedSubfieldSetBeta []SimpleSetItem `json:"chainedSubfieldSetBeta"`
}

type ComplexSetItem struct {
	// +k8s:alpha=+k8s:minimum=10
	Value     int    `json:"value"`
	StringVal string `json:"stringVal"`
}

type ComplexSetItemBeta struct {
	// +k8s:beta=+k8s:minimum=10
	Value     int    `json:"value"`
	StringVal string `json:"stringVal"`
}

type SimpleSetItem struct {
	Value     int    `json:"value"`
	StringVal string `json:"stringVal"`
}
