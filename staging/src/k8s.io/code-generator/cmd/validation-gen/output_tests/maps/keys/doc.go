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
package eachkey

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

// +k8s:validateFalse="type Struct"
type Struct struct {
	TypeMeta int

	// +k8s:validateFalse="field Struct.MapField"
	// +k8s:eachKey=+k8s:validateFalse="field Struct.MapField(keys)"
	MapField map[string]string `json:"mapField"`

	// +k8s:validateFalse="field Struct.MapTypedefField"
	// +k8s:eachKey=+k8s:validateFalse="field Struct.MapTypedefField(keys)"
	MapTypedefField map[UnvalidatedStringType]string `json:"mapTypedefField"`

	// +k8s:validateFalse="field Struct.MapValidatedTypedefField"
	// +k8s:eachKey=+k8s:validateFalse="field Struct.MapValidatedTypedefField(keys)"
	MapValidatedTypedefField map[ValidatedStringType]string `json:"mapValidatedTypedefField"`

	// +k8s:validateFalse="field Struct.MapTypeField"
	// +k8s:eachKey=+k8s:validateFalse="field Struct.MapTypeField(keys)"
	MapTypeField UnvalidatedMapType `json:"mapTypeField"`

	// +k8s:validateFalse="field Struct.ValidatedMapTypeField"
	// +k8s:eachKey=+k8s:validateFalse="field Struct.ValidatedMapTypeField(keys)"
	ValidatedMapTypeField ValidatedMapType `json:"validatedMapTypeField"`
}

// Note: no validations.
type UnvalidatedStringType string

// +k8s:validateFalse="ValidatedStringType"
type ValidatedStringType string

// Note: no validations.
type UnvalidatedMapType map[string]string

// +k8s:validateFalse="ValidatedMapType"
// +k8s:eachKey=+k8s:validateFalse="type ValidatedMapType(keys)"
type ValidatedMapType map[string]string
