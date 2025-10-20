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

// This is a test package for the +k8s:update tag.
package update

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type UpdateTestStruct struct {
	TypeMeta int

	// +k8s:update=NoSet
	StringNoSet string `json:"stringNoSet"`

	// +k8s:update=NoUnset
	StringNoUnset string `json:"stringNoUnset"`

	// +k8s:update=NoModify
	StringNoModify string `json:"stringNoModify"`

	// +k8s:update=NoSet
	// +k8s:update=NoModify
	// +k8s:update=NoUnset
	StringFullyRestricted string `json:"stringFullyRestricted"`

	// +k8s:update=NoModify
	// +k8s:update=NoUnset
	StringSetOnce string `json:"stringSetOnce"`

	// +k8s:update=NoModify
	IntNoModify int `json:"intNoModify"`

	// +k8s:update=NoModify
	Int32NoModify int32 `json:"int32NoModify"`

	// +k8s:update=NoModify
	Int64NoModify int64 `json:"int64NoModify"`

	// +k8s:update=NoModify
	UintNoModify uint `json:"uintNoModify"`

	// +k8s:update=NoModify
	BoolNoModify bool `json:"boolNoModify"`

	// +k8s:update=NoModify
	Float32NoModify float32 `json:"float32NoModify"`

	// +k8s:update=NoModify
	Float64NoModify float64 `json:"float64NoModify"`

	// +k8s:update=NoModify
	ByteNoModify byte `json:"byteNoModify"`

	// +k8s:update=NoModify
	StructNoModify TestStruct `json:"structNoModify"`

	// +k8s:update=NoModify
	NonComparableStructNoModify NonComparableStruct `json:"nonComparableStructNoModify"`

	// Pointer field tests

	// +k8s:update=NoSet
	PointerNoSet *string `json:"pointerNoSet"`

	// +k8s:update=NoUnset
	PointerNoUnset *string `json:"pointerNoUnset"`

	// +k8s:update=NoModify
	PointerNoModify *string `json:"pointerNoModify"`

	// +k8s:update=NoSet
	// +k8s:update=NoModify
	// +k8s:update=NoUnset
	PointerFullyRestricted *string `json:"pointerFullyRestricted"`

	// +k8s:update=NoModify
	IntPointerNoModify *int `json:"intPointerNoModify"`

	// +k8s:update=NoModify
	BoolPointerNoModify *bool `json:"boolPointerNoModify"`

	// +k8s:update=NoModify
	StructPointerNoModify *TestStruct `json:"structPointerNoModify"`

	// Type alias tests

	// +k8s:update=NoModify
	CustomTypeNoModify CustomString `json:"customTypeNoModify"`

	// +k8s:update=NoSet
	CustomTypeNoSet CustomInt `json:"customTypeNoSet"`
}

type TestStruct struct {
	StringField string `json:"stringField"`
	IntField    int    `json:"intField"`
}

// NonComparableStruct contains a slice which makes it non-comparable
type NonComparableStruct struct {
	SliceField []string `json:"sliceField"`
	IntField   int      `json:"intField"`
}

// Custom types to test type aliases
type CustomString string
type CustomInt int
