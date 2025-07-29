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
package listset

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int

	// +k8s:listType=set
	SliceStringField []string `json:"sliceStringField"`

	// +k8s:listType=set
	SliceIntField []int `json:"sliceIntField"`

	// +k8s:listType=set
	SliceComparableField []ComparableStruct `json:"sliceComparableField"`

	// +k8s:listType=set
	SliceNonComparableField []NonComparableStruct `json:"sliceNonComparableField"`

	// +k8s:listType=set
	SliceFalselyComparableField []FalselyComparableStruct `json:"sliceFalselyComparableField"`
}

type ImmutableStruct struct {
	TypeMeta int

	// +k8s:eachVal=+k8s:immutable
	SliceComparableField []ComparableStruct `json:"sliceComparableField"`

	// +k8s:listType=set
	// +k8s:eachVal=+k8s:immutable
	SliceSetComparableField []ComparableStruct `json:"sliceSetComparableField"`

	// +k8s:eachVal=+k8s:immutable
	SliceNonComparableField []NonComparableStruct `json:"sliceNonComparableField"`

	// +k8s:listType=set
	// +k8s:eachVal=+k8s:immutable
	SliceSetNonComparableField []NonComparableStruct `json:"sliceSetNonComparableField"`

	// +k8s:eachVal=+k8s:immutable
	SlicePrimitiveField []int `json:"slicePrimitiveField"`

	// +k8s:listType=set
	// +k8s:eachVal=+k8s:immutable
	SliceSetPrimitiveField []int `json:"sliceSetPrimitiveField"`

	// +k8s:listType=set
	// +k8s:eachVal=+k8s:immutable
	SliceSetFalselyComparableField []FalselyComparableStruct `json:"sliceSetFalselyComparableField"`
}

type ComparableStruct struct {
	StringField string `json:"stringField"`
}

type NonComparableStruct struct {
	SliceField []string `json:"sliceField"`
}

// FalselyComparableStruct contains a pointer field which makes Go's == operator
// only compare the pointers, not the underlying values
type FalselyComparableStruct struct {
	StringPtrField *string `json:"stringPtrField"`
}
