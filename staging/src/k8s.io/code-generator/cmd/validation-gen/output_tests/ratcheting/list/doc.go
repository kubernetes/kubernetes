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

// +k8s:validation-gen=TypeMeta
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme

// This is a test package.
package list

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type StructSlice struct {
	TypeMeta int

	// +k8s:eachVal=+k8s:validateFalse="field SliceField[*]"
	SliceField []S `json:"sliceField"`

	// +k8s:eachVal=+k8s:validateFalse="field TypeDefSliceField[*]"
	TypeDefSliceField MySlice `json:"typedefSliceField"`

	// +k8s:eachVal=+k8s:validateFalse="field SliceStructField[*]"
	SliceStructField []DirectComparableStruct `json:"sliceStructField"`

	// +k8s:eachVal=+k8s:validateFalse="field SliceNonComparableStructField[*]"
	SliceNonComparableStructField []NonDirectComparableStruct `json:"sliceNonComparableStructField"`

	// +k8s:listType=map
	// +k8s:listMapKey=key
	// +k8s:eachVal=+k8s:validateFalse="field SliceStructWithKey[*]"
	SliceStructWithKey []DirectComparableStructWithKey `json:"sliceStructWithKey"`

	// +k8s:listType=map
	// +k8s:listMapKey=key
	// +k8s:eachVal=+k8s:validateFalse="field SliceNonComparableStructWithKey[*]"
	SliceNonComparableStructWithKey []NonComparableStructWithKey `json:"sliceNonComparableStructWithKey"`

	// +k8s:listType=set
	// +k8s:eachVal=+k8s:validateFalse="field SliceSetStructField[*]"
	SliceSetStructField []DirectComparableStruct `json:"sliceSetStructField"`

	// +k8s:listType=set
	// +k8s:eachVal=+k8s:validateFalse="field SliceSetNonComparableStructField[*]"
	SliceSetNonComparableStructField []NonDirectComparableStruct `json:"sliceSetNonComparableStructField"`
}

type S string

type MySlice []int

type DirectComparableStruct struct {
	IntField int `json:"intField"`
}

// +k8s:validateFalse="type NonDirectComparableStruct"
type NonDirectComparableStruct struct {
	IntPtrField *int `json:"intPtrField"`
}

type DirectComparableStructWithKey struct {
	Key string `json:"key"`

	IntField int `json:"intField"`
}

type NonComparableStructWithKey struct {
	Key string `json:"key"`

	IntPtrField *int `json:"intPtrField"`
}
