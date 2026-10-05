/*
Copyright The Kubernetes Authors.

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

// +k8s:validation-gen=TypesWithField=TypeMeta
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme

// This is a test package.
// +k8s:validation-gen-nolint
package customvalidation

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

// +k8s:customValidation
type Struct struct {
	TypeMeta int

	// +k8s:customValidation
	StringField string `json:"stringField"`

	// Combined with a declarative tag: both validations run on this field.
	// +k8s:maxLength=3
	// +k8s:customValidation
	MaxLengthField string `json:"maxLengthField"`

	// StringType is custom-validated wherever it appears via traversal.
	TypedefField      StringType            `json:"typedefField"`
	TypedefPtrField   *StringType           `json:"typedefPtrField,omitempty"`
	TypedefSliceField []StringType          `json:"typedefSliceField,omitempty"`
	TypedefMapField   map[string]StringType `json:"typedefMapField,omitempty"`

	StructField OtherStruct `json:"structField"`
}

// +k8s:customValidation
type StringType string

type OtherStruct struct {
	// +k8s:customValidation
	StringField string `json:"stringField"`
}

// OptionStruct demonstrates custom validation gated behind a feature option.
type OptionStruct struct {
	TypeMeta int

	// +k8s:ifEnabled(FeatureX)=+k8s:customValidation
	StringField string `json:"stringField"`
}

// EachStruct demonstrates custom validation (via the StringType element type)
// coexisting with a declarative per-element check applied by eachVal.
type EachStruct struct {
	TypeMeta int

	// +k8s:eachVal=+k8s:maxLength=3
	SliceField []StringType `json:"sliceField"`
}
