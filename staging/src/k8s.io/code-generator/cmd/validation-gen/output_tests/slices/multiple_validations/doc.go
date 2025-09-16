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
package multiplevalidations

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

// +k8s:validateFalse="type Struct #1"
// +k8s:validateFalse="type Struct #2"
type Struct struct {
	TypeMeta int

	// +k8s:validateFalse="field Struct.ListField #1"
	// +k8s:validateFalse="field Struct.ListField #2"
	// +k8s:eachVal=+k8s:validateFalse="field Struct.ListField[*] #1"
	// +k8s:eachVal=+k8s:validateFalse="field Struct.ListField[*] #2"
	ListField []string `json:"listField"`

	UnvalidatedListField []string `json:"UnvalidatedListField"`
}
