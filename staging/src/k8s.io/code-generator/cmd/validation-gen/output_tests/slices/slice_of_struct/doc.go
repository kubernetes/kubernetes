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
package sliceofstruct

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

// +k8s:validateFalse="type Struct"
type Struct struct {
	TypeMeta int

	// +k8s:validateFalse="field Struct.ListField"
	// +k8s:eachVal=+k8s:validateFalse="field Struct.ListField[*]"
	ListField []OtherStruct `json:"listField"`

	// +k8s:validateFalse="field Struct.ListTypedefField"
	// +k8s:eachVal=+k8s:validateFalse="field Struct.ListTypedefField[*]"
	ListTypedefField []OtherTypedefStruct `json:"listTypedefField"`

	UnvalidatedListField []OtherStruct `json:"UnvalidatedListField"`
}

// +k8s:validateFalse="type OtherStruct"
type OtherStruct struct{}

// +k8s:validateFalse="type OtherTypedefStruct"
type OtherTypedefStruct OtherStruct
