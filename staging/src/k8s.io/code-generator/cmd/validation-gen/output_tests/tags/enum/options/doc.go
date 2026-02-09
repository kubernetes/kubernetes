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
package options

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

	EnumWithExcludeField    EnumWithExclude  `json:"enumWithExcludeField"`
	EnumWithExcludePtrField *EnumWithExclude `json:"enumWithExcludePtrField"`
}

type ConditionalStruct struct {
	TypeMeta int

	ConditionalEnumField    ConditionalEnum  `json:"conditionalEnumField"`
	ConditionalEnumPtrField *ConditionalEnum `json:"conditionalEnumPtrField"`
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

// +k8s:enum
type EnumWithExclude string

const (
	EnumWithExclude1 EnumWithExclude = "enumWithExclude1"

	// +k8s:enumExclude
	EnumWithExclude2 EnumWithExclude = "enumWithExclude2"
)

// +k8s:enum
type ConditionalEnum string

const (
	// +k8s:ifEnabled(FeatureA)=+k8s:enumExclude
	ConditionalA ConditionalEnum = "A"

	// +k8s:ifDisabled(FeatureB)=+k8s:enumExclude
	ConditionalB ConditionalEnum = "B"

	// This value is always included.
	ConditionalC ConditionalEnum = "C"

	// +k8s:ifEnabled(FeatureA)=+k8s:enumExclude
	// +k8s:ifEnabled(FeatureB)=+k8s:enumExclude
	ConditionalD ConditionalEnum = "D"

	// +k8s:ifDisabled(FeatureC)=+k8s:enumExclude
	// +k8s:ifDisabled(FeatureD)=+k8s:enumExclude
	ConditionalE ConditionalEnum = "E"

	// +k8s:ifDisabled(FeatureC)=+k8s:enumExclude
	// +k8s:ifEnabled(FeatureD)=+k8s:enumExclude
	ConditionalF ConditionalEnum = "F"
)
