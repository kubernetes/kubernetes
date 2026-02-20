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

package listmapitem

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type MapItem struct {
	Key   string `json:"key"`
	Value int    `json:"value"`
}

type InnerItem struct {
	Value int `json:"value"`
}

type ListMapItemStruct struct {
	TypeMeta int

	// Case: Standard Item
	// +k8s:listType=map
	// +k8s:listMapKey=key
	// +k8s:item(key: "foo")=+k8s:subfield(value)=+k8s:minimum=10
	StandardItem []MapItem `json:"standardItem"`

	// Case: Alpha Item tag
	// +k8s:listType=map
	// +k8s:listMapKey=key
	// +k8s:alpha=+k8s:item(key: "foo")=+k8s:subfield(value)=+k8s:minimum=10
	AlphaItemTag []MapItem `json:"alphaItemTag"`

	// Case: Alpha validation
	// +k8s:listType=map
	// +k8s:listMapKey=key
	// +k8s:item(key: "foo")=+k8s:subfield(value)=+k8s:alpha=+k8s:minimum=10
	AlphaValidation []MapItem `json:"alphaValidation"`

	// Case: Double Alpha
	// +k8s:listType=map
	// +k8s:listMapKey=key
	// +k8s:alpha=+k8s:item(key: "foo")=+k8s:subfield(value)=+k8s:alpha=+k8s:minimum=10
	DoubleAlpha []MapItem `json:"doubleAlpha"`

	// Case: Beta Item tag
	// +k8s:listType=map
	// +k8s:listMapKey=key
	// +k8s:beta=+k8s:item(key: "foo")=+k8s:subfield(value)=+k8s:minimum=10
	BetaItemTag []MapItem `json:"betaItemTag"`

	// Case: Beta validation
	// +k8s:listType=map
	// +k8s:listMapKey=key
	// +k8s:item(key: "foo")=+k8s:subfield(value)=+k8s:beta=+k8s:minimum=10
	BetaValidation []MapItem `json:"betaValidation"`

	// Case: Double Beta
	// +k8s:listType=map
	// +k8s:listMapKey=key
	// +k8s:beta=+k8s:item(key: "foo")=+k8s:subfield(value)=+k8s:beta=+k8s:minimum=10
	DoubleBeta []MapItem `json:"doubleBeta"`
}
