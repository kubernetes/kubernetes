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

// +k8s:validation-gen=*
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme

// This is a test package.
package opaque

import (
	"k8s.io/code-generator/cmd/validation-gen/testscheme"
)

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int

	// +k8s:validateFalse="field Struct.StructField"
	StructField OtherStruct `json:"structField"`

	// +k8s:validateFalse="field Struct.StructPtrField"
	// +k8s:required
	StructPtrField *OtherStruct `json:"structPtrField"`

	// +k8s:validateFalse="field Struct.OpaqueStructField"
	// +k8s:opaqueType
	OpaqueStructField OtherStruct `json:"opaqueStructField"`

	// +k8s:validateFalse="field Struct.OpaqueStructPtrField"
	// +k8s:required
	// +k8s:opaqueType
	OpaqueStructPtrField *OtherStruct `json:"opaqueStructPtrField"`

	// +k8s:validateFalse="field Struct.SliceOfStructField"
	// +k8s:eachVal=+k8s:validateFalse="field Struct.SliceOfStructField vals"
	SliceOfStructField []OtherStruct `json:"sliceOfStructField"`

	// +k8s:validateFalse="field Struct.SliceOfOpaqueStructField"
	// +k8s:eachVal=+k8s:validateFalse="field Struct.SliceOfOpaqueStructField vals"
	// +k8s:eachVal=+k8s:opaqueType
	SliceOfOpaqueStructField []OtherStruct `json:"sliceOfOpaqueStructField"`

	// +k8s:validateFalse="field Struct.ListMapOfStructField"
	// +k8s:eachVal=+k8s:validateFalse="field Struct.ListMapOfStructField vals"
	// +k8s:listType=map
	// +k8s:listMapKey=stringField
	ListMapOfStructField []OtherStruct `json:"listMapOfStructField"`

	// +k8s:validateFalse="field Struct.ListMapOfOpaqueStructField"
	// +k8s:eachVal=+k8s:validateFalse="field Struct.ListMapOfOpaqueStructField vals"
	// +k8s:listType=map
	// +k8s:listMapKey=stringField
	// +k8s:eachVal=+k8s:opaqueType
	ListMapOfOpaqueStructField []OtherStruct `json:"listMapOfOpaqueStructField"`

	// +k8s:validateFalse="field Struct.MapOfStringToStructField"
	// +k8s:eachKey=+k8s:validateFalse="field Struct.MapOfStringToStructField keys"
	// +k8s:eachVal=+k8s:validateFalse="field Struct.MapOfStringToStructField vals"
	MapOfStringToStructField map[OtherString]OtherStruct `json:"mapOfStringToStructField"`

	// +k8s:validateFalse="field Struct.MapOfStringToOpaqueStructField"
	// +k8s:eachKey=+k8s:validateFalse="field Struct.MapOfStringToOpaqueStructField keys"
	// +k8s:eachVal=+k8s:validateFalse="field Struct.MapOfStringToOpaqueStructField vals"
	// +k8s:eachKey=+k8s:opaqueType
	// +k8s:eachVal=+k8s:opaqueType
	MapOfStringToOpaqueStructField map[OtherString]OtherStruct `json:"mapOfStringToOpaqueStructField"`
}

// +k8s:validateFalse="type OtherStruct"
type OtherStruct struct {
	// +k8s:validateFalse="field OtherStruct.StringField"
	StringField string `json:"stringField"`
}

// +k8s:validateFalse="type OtherString"
type OtherString string

// TODO: the validateFalse test fixture doesn't handle map and slice types, and
// fixing it requires fixing randfill.  That is a tomorrow problem.  For now, the
// following types have been tested to generate correct code with
// +k8s:opaqueType.

// +k8s:validateTrue="type TypedefSliceOther"
// +k8s:eachVal=+k8s:opaqueType
type TypedefSliceOther []OtherStruct

// +k8s:validateTrue="type TypedefMapOther"
// +k8s:eachKey=+k8s:opaqueType
// +k8s:eachVal=+k8s:opaqueType
type TypedefMapOther map[OtherString]OtherStruct
