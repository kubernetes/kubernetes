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
// +k8s:validation-gen-test-fixture=validateFalse

// This is a test package.
package typeargs

import (
	"k8s.io/code-generator/cmd/validation-gen/output_tests/primitives"
	"k8s.io/code-generator/cmd/validation-gen/testscheme"
)

var localSchemeBuilder = testscheme.New()

// Explicitly set the type-arg to prove it renders properly.
// NOTE: because of how validation code is generated, these must always be
// pointers, because that is what gets passed around.
type T1 struct {
	TypeMeta int

	// +k8s:validateFalse={"typeArg":"k8s.io/code-generator/cmd/validation-gen/output_tests/primitives.T1", "msg":"T1.S1"}
	S1 *primitives.T1 `json:"s1"`
	// +k8s:validateFalse={"typeArg":"k8s.io/code-generator/cmd/validation-gen/output_tests/primitives.T1", "msg":"PT1.PS1"}
	PS1 *primitives.T1 `json:"ps1"`

	// +k8s:validateFalse={"typeArg":"k8s.io/code-generator/cmd/validation-gen/output_tests/type_args.E1", "msg":"T1.E1"}
	E1 E1 `json:"e1"`
	// +k8s:validateTrue={"typeArg":"k8s.io/code-generator/cmd/validation-gen/output_tests/type_args.E1", "msg":"T1.PE1"}
	PE1 *E1 `json:"pe1"`

	// +k8s:validateFalse={"typeArg":"int", "msg":"T1.I1"}
	I1 int `json:"i1"`
	// +k8s:validateTrue={"typeArg":"int", "msg":"T1.PI1"}
	PI1 *int `json:"pi1"`
}

// +k8s:validateFalse={"msg": "type E1"}
type E1 string
