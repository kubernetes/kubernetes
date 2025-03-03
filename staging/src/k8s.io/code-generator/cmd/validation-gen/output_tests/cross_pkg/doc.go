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

// +k8s:validation-gen=*
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme
// +k8s:validation-gen-test-fixture=validateFalse

//nolint:unused

// This is a test package.
package crosspkg

import (
	"k8s.io/code-generator/cmd/validation-gen/output_tests/_codegenignore/other"
	"k8s.io/code-generator/cmd/validation-gen/output_tests/primitives"
	"k8s.io/code-generator/cmd/validation-gen/output_tests/typedefs"
	"k8s.io/code-generator/cmd/validation-gen/testscheme"
)

var localSchemeBuilder = testscheme.New()

type T1 struct {
	TypeMeta int

	// +k8s:validateTrue="field T1.PrimitivesT1"
	PrimitivesT1 primitives.T1 `json:"primitivest1"`
	// +k8s:validateTrue="field T1.PrimitivesT1Ptr"
	PrimitivesT1Ptr *primitives.T1 `json:"primitivest1Ptr"`
	// +k8s:validateTrue="field T1.PrimitivesT2"
	PrimitivesT2 primitives.T2 `json:"primitivest2"`
	// +k8s:validateTrue="field T1.PrimitivesT2Ptr"
	PrimitivesT2Ptr *primitives.T1 `json:"primitivest2Ptr"`
	// +k8s:validateTrue="field T1.PrimitivesT3"
	PrimitivesT3 primitives.T3 `json:"primitivest3"`
	// +k8s:validateTrue="field T1.PrimitivesT3Ptr"
	PrimitivesT3Ptr *primitives.T1 `json:"primitivest3Ptr"`
	// T4 and T5 are not root types in that pkg and are not linked into any
	// root type's transitive graph, so they have no functions.

	// +k8s:validateTrue="field T1.TypedefsE1"
	TypedefsE1 typedefs.E1 `json:"typedefse1"`
	// +k8s:validateTrue="field T1.TypedefsE1Ptr"
	TypedefsE1Ptr *typedefs.E1 `json:"typedefse1Ptr"`
	// +k8s:validateTrue="field T1.TypedefsE2"
	TypedefsE2 typedefs.E2 `json:"typedefse2"`
	// +k8s:validateTrue="field T1.TypedefsE2Ptr"
	TypedefsE2Ptr *typedefs.E2 `json:"typedefse2Ptr"`
	// +k8s:validateTrue="field T1.TypedefsE3"
	TypedefsE3 typedefs.E3 `json:"typedefse3"`
	// +k8s:validateTrue="field T1.TypedefsE3Ptr"
	TypedefsE3Ptr *typedefs.E3 `json:"typedefse3Ptr"`
	// +k8s:validateTrue="field T1.TypedefsE4"
	TypedefsE4 typedefs.E4 `json:"typedefse4"`
	// +k8s:validateTrue="field T1.TypedefsE4Ptr"
	TypedefsE4Ptr *typedefs.E4 `json:"typedefse4Ptr"`

	// +k8s:validateTrue="field T1.OtherString"
	// +k8s:opaqueType
	OtherString other.StringType `json:"otherString"`
	// +k8s:validateTrue="field T1.OtherStringPtr"
	// +k8s:opaqueType
	OtherStringPtr *other.StringType `json:"otherStringPtr"`
	// +k8s:validateTrue="field T1.OtherInt"
	// +k8s:opaqueType
	OtherInt other.IntType `json:"otherInt"`
	// +k8s:validateTrue="field T1.OtherIntPtr"
	// +k8s:opaqueType
	OtherIntPtr *other.IntType `json:"otherIntPtr"`
	// +k8s:validateTrue="field T1.OtherStruct"
	// +k8s:opaqueType
	OtherStruct other.StructType `json:"otherStruct"`
	// +k8s:validateTrue="field T1.OtherStructPtr"
	// +k8s:opaqueType
	OtherStructPtr *other.StructType `json:"otherStructPtr"`

	// +k8s:validateTrue="field T1.SliceOfOtherStruct"
	// +k8s:eachVal=+k8s:validateTrue="field T1.SliceOfOtherStruct values"
	// +k8s:eachVal=+k8s:opaqueType
	SliceOfOtherStruct []other.StructType `json:"sliceOfOtherStruct"`
	// +k8s:validateTrue="field T1.SliceOfOtherStructPtr"
	// +k8s:eachVal=+k8s:validateTrue="field T1.SliceOfOtherStructPtr values"
	// +k8s:eachVal=+k8s:opaqueType
	SliceOfOtherStructPtr []*other.StructType `json:"sliceOfOtherStructPtr"`

	// +k8s:validateTrue="field T1.ListMapOfOtherStruct"
	// +k8s:eachVal=+k8s:validateTrue="field T1.SliceOfOtherStruct values"
	// +k8s:listType=map
	// +k8s:listMapKey=stringField
	// +k8s:eachVal=+k8s:opaqueType
	ListMapOfOtherStruct []other.StructType `json:"listMapOfOtherStruct"`
	// +k8s:validateTrue="field T1.ListMapOfOtherStructPtr"
	// +k8s:eachVal=+k8s:validateTrue="field T1.SliceOfOtherStructPtr values"
	// +k8s:listType=map
	// +k8s:listMapKey=stringField
	// +k8s:eachVal=+k8s:opaqueType
	ListMapOfOtherStructPtr []*other.StructType `json:"listMapOfOtherStructPtr"`

	// +k8s:validateTrue="field T1.MapOfOtherStringToOtherStruct"
	// +k8s:eachKey=+k8s:validateTrue="field T1.MapOfOtherStringToOtherStruct keys"
	// +k8s:eachVal=+k8s:validateTrue="field T1.MapOfOtherStringToOtherStruct values"
	// +k8s:eachKey=+k8s:opaqueType
	// +k8s:eachVal=+k8s:opaqueType
	MapOfOtherStringToOtherStruct map[other.StringType]other.StructType `json:"mapOfOtherStringToOtherStruct"`
	// +k8s:validateTrue="field T1.MapOfOtherStringToOtherStructPtr"
	// +k8s:eachKey=+k8s:validateTrue="field T1.MapOfOtherStringToOtherStructPtr keys"
	// +k8s:eachVal=+k8s:validateTrue="field T1.MapOfOtherStringToOtherStructPtr values"
	// +k8s:eachKey=+k8s:opaqueType
	// +k8s:eachVal=+k8s:opaqueType
	MapOfOtherStringToOtherStructPtr map[other.StringType]*other.StructType `json:"mapOfOtherStringToOtherStructPtr"`
}

// TODO: the validateFalse test fixture doesn't handle map and slice types, and
// fixing it requires fixing gofuzz.  That is a tomorrow problem.  For now, the
// following types have been tested to fail without +k8s:opaqueType.

/*
// +k8s:validateTrue="type TypedefSliceOther"
// +k8s:eachVal=+k8s:opaqueType
type TypedefSliceOther []other.StructType

// +k8s:validateTrue="type TypedefMapOther"
// +k8s:eachKey=+k8s:opaqueType
// +k8s:eachVal=+k8s:opaqueType
type TypedefMapOther map[other.StringType]other.StructType
*/
