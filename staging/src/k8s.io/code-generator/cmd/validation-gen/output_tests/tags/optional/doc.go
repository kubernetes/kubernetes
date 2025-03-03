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
package optional

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int

	// +k8s:optional
	// +k8s:validateFalse="field Struct.StringField"
	StringField string `json:"stringField"`

	// +k8s:optional
	// +k8s:validateFalse="field Struct.StringPtrField"
	StringPtrField *string `json:"stringPtrField"`

	// +k8s:optional
	// +k8s:validateFalse="field Struct.OtherStructField"
	OtherStructField OtherStruct `json:"otherStructField"`

	// +k8s:optional
	// +k8s:validateFalse="field Struct.OtherStructPtrField"
	OtherStructPtrField *OtherStruct `json:"otherStructPtrField"`

	// +k8s:optional
	// +k8s:validateFalse="field Struct.SliceField"
	SliceField []string `json:"sliceField"`

	// +k8s:optional
	// +k8s:validateFalse="field Struct.MapField"
	MapField map[string]string `json:"mapField"`
}

// +k8s:validateFalse="type OtherStruct"
type OtherStruct struct{}
