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
package eachval

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type StructWithMaps struct {
	TypeMeta int

	// +k8s:eachVal=+k8s:validateFalse="field MapTest.MapPrimitiveField[*]"
	MapPrimitiveField map[string]string `json:"mapPrimitiveField"`

	// +k8s:eachVal=+k8s:validateFalse="field MapTest.MapTypedefField[*]"
	MapTypedefField map[string]StringType `json:"mapTypedefField"`

	// +k8s:eachVal=+k8s:validateFalse="field MapTest.MapComparableStructField[*]"
	MapComparableStructField map[string]ComparableStruct `json:"mapComparableStructField"`

	// +k8s:eachVal=+k8s:validateFalse="field MapTest.MapNonComparableStructField[*]"
	MapNonComparableStructField map[string]NonComparableStruct `json:"mapNonComparableStructField"`
}

type StringType string

type ComparableStruct struct {
	IntField int `json:"intField"`
}

type NonComparableStruct struct {
	IntPtrField *int `json:"intPtrField"`
}
