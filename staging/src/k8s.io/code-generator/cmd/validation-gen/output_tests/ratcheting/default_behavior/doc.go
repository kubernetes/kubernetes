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
package defaultbehavior

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type StructPrimitive struct {
	TypeMeta int

	// +k8s:validateFalse="field intField"
	IntField int `json:"intField"`

	// +k8s:optional
	// +k8s:validateFalse="field intPtrField"
	IntPtrField *int `json:"intPtrField"`
}

type StructSlice struct {
	TypeMeta int

	// +k8s:validateFalse="field sliceField"
	SliceField []S `json:"sliceField"`

	// +k8s:validateFalse="field typedefSliceField"
	TypeDefSliceField MySlice `json:"typedefSliceField"`
}

type StructMap struct {
	TypeMeta int

	// +k8s:validateFalse="field mapKeyField"
	MapKeyField map[S]string `json:"mapKeyField"`

	// +k8s:validateFalse="field mapValueField"
	MapValueField map[string]S `json:"mapValueField"`

	// +k8s:validateFalse="field aliasMapKeyTypeField"
	AliasMapKeyTypeField AliasMapKeyType `json:"aliasMapKeyTypeField"`

	// +k8s:validateFalse="field aliasMapValueTypeField"
	AliasMapValueTypeField AliasMapValueType `json:"aliasMapValueTypeField"`
}

type StructStruct struct {
	TypeMeta int

	// +k8s:validateFalse="field directComparableStructField"
	DirectComparableStructField DirectComparableStruct `json:"directComparableStructField"`

	// +k8s:validateFalse="field nonDirectComparableStructField"
	NonDirectComparableStructField NonDirectComparableStruct `json:"nonDirectComparableStructField"`

	// +k8s:validateFalse="field directComparableStructPtrField"
	DirectComparableStructPtr *DirectComparableStruct `json:"directComparableStructPtrField"`

	// +k8s:validateFalse="field nonDirectComparableStructPtrField"
	NonDirectComparableStructPtr *NonDirectComparableStruct `json:"nonDirectComparableStructPtrField"`

	// +k8s:validateFalse="field DirectComparableStruct"
	DirectComparableStruct

	// +k8s:validateFalse="field NonDirectComparableStruct"
	NonDirectComparableStruct
}

type StructEmbedded struct {
	TypeMeta int
	// +k8s:validateFalse="field DirectComparableStruct"
	DirectComparableStruct `json:"directComparableStruct"`

	// +k8s:validateFalse="field NonDirectComparableStruct"
	NonDirectComparableStruct `json:"nonDirectComparableStruct"`

	// +k8s:validateFalse="field NestedDirectComparableStructField"
	NestedDirectComparableStructField NestedDirectComparableStruct `json:"nestedDirectComparableStructField"`

	// +k8s:validateFalse="field NestedNonDirectComparableStructField"
	NestedNonDirectComparableStructField NestedNonDirectComparableStruct `json:"nestedNonDirectComparableStructField"`
}

// +k8s:validateFalse="type TypeDefStruct"
type TypeDefStruct struct{}

// +k8s:validateFalse="type MySlice"
type MySlice []int

// +k8s:validateFalse="type S"
type S string

// +k8s:validateFalse="type MapKeyType"
type AliasMapKeyType MapKeyType

// +k8s:validateFalse="type MapValueType"
type AliasMapValueType MapValueType

// no validation
type MapKeyType map[S]string

// no validation
type MapValueType map[string]S

// +k8s:validateFalse="type DirectComparableStruct"
type DirectComparableStruct struct {
	// +k8s:validateFalse="field intField"
	IntField int `json:"intField"`
}

// +k8s:validateFalse="type NonDirectComparableStruct"
type NonDirectComparableStruct struct {
	// +k8s:validateFalse="field intField"
	IntPtrField *int `json:"intPtrField"`
}

// +k8s:validateFalse="type NestedDirectComparableStruct"
type NestedDirectComparableStruct struct {
	// +k8s:validateFalse="field intField"
	DirectComparableStructField DirectComparableStruct `json:"directComparableStructField"`
}

// +k8s:validateFalse="type NestedNonDirectComparableStruct"
type NestedNonDirectComparableStruct struct {
	// +k8s:validateFalse="field intField"
	NonDirectComparableStructField NonDirectComparableStruct `json:"nonDirectComparableStructField"`
}
