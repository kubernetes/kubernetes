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
package basics

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

// MyObject is a test struct.
// +k8s:validation:Required
type MyObject struct {
	TypeMeta int
	// +k8s:optional
	// +k8s:format=k8s-uuid
	// +k8s:declarativeValidationNative
	UUIDField string `json:"uuidField"`

	// +k8s:optional
	// +k8s:format=k8s-uuid
	UUIDFieldWithoutDV string `json:"uuidFieldWithoutDV"`

	// +k8s:optional
	// +k8s:format=k8s-uuid
	// +k8s:declarativeValidationNative
	UUIDPtrField *string `json:"uuidPtrField"`

	// +k8s:optional
	// +k8s:format=k8s-uuid
	UUIDPtrFieldWithoutDV *string `json:"uuidPtrFieldWithoutDV"`

	// Note: no validation here
	// +k8s:declarativeValidationNative
	UUIDTypedefField UUIDString `json:"uuidTypedefField"`

	UUIDTypedefFieldWithoutDV UUIDString `json:"uuidTypedefFieldWithoutDV"`

	// +k8s:declarativeValidationNative
	// +k8s:maxLength=60
	// +k8s:required
	FieldForLength string `json:"fieldForLength"`

	// +k8s:maxLength=60
	// +k8s:required
	FieldForLengthWithoutDV string `json:"fieldForLengthWithoutDV"`

	// +k8s:declarativeValidationNative
	StableTypeField StableType `json:"stableTypeField"`

	StableTypeFieldWithoutDV StableType `json:"stableTypeFieldWithoutDV"`

	// +k8s:declarativeValidationNative
	StableTypeFieldPointer *StableType `json:"stableTypeFieldPointer"`

	StableTypeFieldPointerWithoutDV *StableType `json:"stableTypeFieldPointerWithoutDV"`

	// +k8s:declarativeValidationNative
	// +k8s:maxItems=5
	StableTypeSlice []StableType `json:"stableTypeSlice"`

	// +k8s:maxItems=5
	StableTypeSliceWithoutDV []StableType `json:"stableTypeSliceWithoutDV"`

	// +k8s:declarativeValidationNative
	// +k8s:listType=set
	SetList []string `json:"setList"`

	// +k8s:listType=set
	SetListWithoutDV []string `json:"setListWithoutDV"`

	// +k8s:declarativeValidationNative
	NestedStable NestedStableType `json:"nestedStable"`

	NestedStableWithoutDV NestedStableType `json:"nestedStableWithoutDV"`

	// +k8s:subfield(innerField)=+k8s:maxLength=5
	// +k8s:declarativeValidationNative
	SubfieldTest StableType `json:"subfieldTest"`

	// +k8s:subfield(innerField)=+k8s:maxLength=5
	SubfieldTestWithoutDV StableType `json:"subfieldTestWithoutDV"`
}

// +k8s:format=k8s-uuid
type UUIDString string

// Type-level case
type StableType struct {
	// +k8s:required
	// +k8s:maxLength=10
	InnerField string `json:"innerField"`
}

type NestedStableType struct {
	// +k8s:declarativeValidationNative
	NestedField StableType `json:"nestedField"`

	NestedFieldWithoutDV StableType `json:"nestedFieldWithoutDV"`
}
