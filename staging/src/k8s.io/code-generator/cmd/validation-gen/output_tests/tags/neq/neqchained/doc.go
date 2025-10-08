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

// This is a test package for complex neq compositions.
package neqchained

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int

	// +k8s:subfield(stringField)=+k8s:neq="disallowed-subfield"
	StructField InnerStruct `json:"structField"`

	// +k8s:optional
	// +k8s:subfield(stringField)=+k8s:neq="disallowed-subfield-ptr"
	StructPtrField *InnerStruct `json:"structPtrField"`

	// +k8s:eachVal=+k8s:neq="disallowed-slice"
	StringSliceField []string `json:"stringSliceField"`

	// +k8s:eachVal=+k8s:neq="disallowed-map-val"
	StringMapField map[string]string `json:"stringMapField"`

	// +k8s:eachKey=+k8s:neq="disallowed-key"
	StringMapKeyField map[string]string `json:"stringMapKeyField"`

	ValidatedSliceField ValidatedStringSlice `json:"validatedSliceField"`

	ValidatedStructField ValidatedInnerStruct `json:"validatedStructField"`
}

type InnerStruct struct {
	StringField string `json:"stringField"`
}

// +k8s:eachVal=+k8s:neq="disallowed-typedef"
type ValidatedStringSlice []string

// +k8s:subfield(stringField)=+k8s:neq="disallowed-typedef-struct"
type ValidatedInnerStruct InnerStruct
