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
package enum

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int

	Enum0Field    Enum0  `json:"enum0Field"`
	Enum0PtrField *Enum0 `json:"enum0PtrField"`

	Enum1Field    Enum1  `json:"enum1Field"`
	Enum1PtrField *Enum1 `json:"enum1PtrField"`

	Enum2Field    Enum2  `json:"enum2Field"`
	Enum2PtrField *Enum2 `json:"enum2PtrField"`

	NotEnumField    NotEnum  `json:"notEnumField"`
	NotEnumPtrField *NotEnum `json:"notEnumPtrField"`
}

// +k8s:enum
type Enum0 string // Note: this enum has no values

// +k8s:enum
type Enum1 string // Note: this enum has 1 value

const (
	E1V1 Enum1 = "e1v1"
)

// +k8s:enum
type Enum2 string // Note: this enum has 2 values

const (
	E2V1 Enum2 = "e2v1"
	E2V2 Enum2 = "e2v2"
)

// Note: this is not an enum because the const values are of type Enum2, and
// because go elides intermediate typedefs (this is modelled as "NotEnum" ->
// "string" in the AST).
type NotEnum Enum2
