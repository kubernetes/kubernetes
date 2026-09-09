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

// +k8s:validation-gen=TypesWithField=TypeMeta
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme
// +k8s:validation-gen-deep-equal-func=k8s.io/code-generator/cmd/validation-gen/output_tests/_codegenignore/other.DeepEqual

// This is a test package.  Unlike deep_equal_func, which names a function in
// its own package, this one names a function in another package, so the
// generated code must emit an import for it.
// +k8s:validation-gen-nolint
package deepequalfunccrosspkg

import (
	"k8s.io/code-generator/cmd/validation-gen/testscheme"
)

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int

	// +k8s:validateFalse="field Struct.NonComparableField"
	NonComparableField NonComparableStruct `json:"nonComparableField"`
}

// +k8s:validateFalse="type NonComparableStruct"
type NonComparableStruct struct {
	// Ptr makes it not direct-comparable
	Ptr *int `json:"ptr"`
}
