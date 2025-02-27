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

// This is a test package.
package typedeftomap

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

// Note: no validation here
type UnvalidatedType map[string]string

// +k8s:eachVal=+k8s:validateFalse="type MapType[*]"
type MapType map[string]string

// Note: no validation here
type UnvalidatedPtrType map[string]*string

// +k8s:eachVal=+k8s:validateFalse="type MapPtrType[*]"
type MapPtrType map[string]*string

// +k8s:validateFalse="type StringType"
type StringType string

// +k8s:eachVal=+k8s:validateFalse="type MapTypedefType[*]"
type MapTypedefType map[string]StringType

// +k8s:validateFalse="type Struct"
type Struct struct {
	TypeMeta int

	// +k8s:eachVal=+k8s:validateFalse="field Struct.MapField[*]"
	MapField MapType `json:"mapField"`

	// +k8s:eachVal=+k8s:validateFalse="field Struct.MapPtrField[*]"
	MapPtrField MapPtrType `json:"mapPtrField"`

	// +k8s:eachVal=+k8s:validateFalse="field Struct.MapTypedefField[*]"
	MapTypedefField MapTypedefType `json:"mapTypedefField"`
}
