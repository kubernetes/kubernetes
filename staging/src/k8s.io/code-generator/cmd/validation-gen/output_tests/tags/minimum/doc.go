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
// +k8s:validation-gen-nolint
package minimum

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type BasicStruct struct {
	TypeMeta int

	// +k8s:minimum=1
	IntField int `json:"intField"`
	// +k8s:minimum=1
	IntPtrField *int `json:"intPtrField"`

	// "int8" becomes "byte" somewhere in gengo.  We don't need it so just skip it.

	// +k8s:minimum=1
	Int16Field int16 `json:"int16Field"`
	// +k8s:minimum=1
	Int32Field int32 `json:"int32Field"`
	// +k8s:minimum=1
	Int64Field int64 `json:"int64Field"`

	// +k8s:minimum=1
	UintField uint `json:"uintField"`
	// +k8s:minimum=1
	UintPtrField *uint `json:"uintPtrField"`

	// +k8s:minimum=1
	Uint16Field uint16 `json:"uint16Field"`
	// +k8s:minimum=1
	Uint32Field uint32 `json:"uint32Field"`
	// +k8s:minimum=1
	Uint64Field uint64 `json:"uint64Field"`

	TypedefField    IntType  `json:"typedefField"`
	TypedefPtrField *IntType `json:"typedefPtrField"`
}

type OptionalStruct struct {
	TypeMeta int

	// +k8s:optional
	// +k8s:minimum=1
	OptionalIntField int `json:"optionalIntField"`

	// +k8s:optional
	// +k8s:minimum=1
	OptionalIntPtrField *int `json:"optionalIntPtrField"`

	// +k8s:optional
	OptionalTypedefField IntType `json:"optionalTypedefField"`

	// +k8s:optional
	OptionalTypedefPtrField *IntType `json:"optionalTypedefPtrField"`
}

type RequiredStruct struct {
	TypeMeta int

	// +k8s:required
	// +k8s:minimum=1
	RequiredIntField int `json:"requiredIntField"`

	// +k8s:required
	// +k8s:minimum=1
	RequiredIntPtrField *int `json:"requiredIntPtrField"`

	// +k8s:required
	RequiredTypedefField IntType `json:"requiredTypedefField"`

	// +k8s:required
	RequiredTypedefPtrField *IntType `json:"requiredTypedefPtrField"`
}

type NegativeMinimumStruct struct {
	TypeMeta int

	// +k8s:minimum=-10
	NegativeMinimumField int `json:"negativeMinimumField"`

	// +k8s:minimum=-10
	NegativeMinimumPtrField *int `json:"negativeMinimumPtrField"`

	// +k8s:optional
	// +k8s:minimum=-10
	OptionalNegativeMinimumField int `json:"optionalNegativeMinimumField"`

	// +k8s:optional
	// +k8s:minimum=-10
	OptionalNegativeMinimumPtrField *int `json:"optionalNegativeMinimumPtrField"`

	// +k8s:required
	// +k8s:minimum=-10
	RequiredNegativeMinimumField int `json:"requiredNegativeMinimumField"`

	// +k8s:required
	// +k8s:minimum=-10
	RequiredNegativeMinimumPtrField *int `json:"requiredNegativeMinimumPtrField"`
}

// +k8s:minimum=1
type IntType int
