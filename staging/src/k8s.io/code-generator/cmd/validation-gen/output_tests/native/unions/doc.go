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

// +k8s:deepcopy-gen=package
// +k8s:validation-gen=TypeMeta
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme

// This is a test package.
package unions

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int
	// +k8s:optional
	// +k8s:unionMember
	// +k8s:declarativeValidationNative
	UnionField1 *int `json:"unionField1,omitempty"`
	// +k8s:optional
	// +k8s:unionMember
	// +k8s:declarativeValidationNative
	UnionField2 *int `json:"unionField2,omitempty"`
}

type ListStruct struct {
	TypeMeta int
	// +k8s:listType=map
	// +k8s:listMapKey=type
	// +k8s:declarativeValidationNative
	// +k8s:item(type: "a")=+k8s:unionMember
	// +k8s:item(type: "b")=+k8s:unionMember
	Items []UnionItem `json:"items"`
}

type UnionItem struct {
	Type string `json:"type"`
	A    *int   `json:"a,omitempty"`
	B    *int   `json:"b,omitempty"`
}
