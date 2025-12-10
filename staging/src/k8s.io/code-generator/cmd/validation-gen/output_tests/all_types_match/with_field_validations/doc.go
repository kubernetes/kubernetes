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

// Note: this selects all types in the package.
// +k8s:validation-gen=*
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme
// +k8s:validation-gen-test-fixture=validateFalse

// This is a test package.
package withfieldvalidations

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type T1 struct {
	// +k8s:validateFalse="field T1.S"
	S string `json:"s"`
	// +k8s:validateFalse="field T1.T2"
	T2 T2 `json:"t2"`
	// +k8s:validateFalse="field T1.T3"
	T3 T3 `json:"t3"`
}

// Note: this has validations and is linked into T1.
type T2 struct {
	// +k8s:validateFalse="field T2.S"
	S string `json:"s"`
}

// Note: this has no validations and is linked into T1.
type T3 struct {
	S string `json:"s"`
}

// Note: this has validations and is not linked into T1.
type T4 struct {
	// +k8s:validateFalse="field T4.S"
	S string `json:"s"`
}

// Note: this has no validations and is not linked into T1.
type T5 struct {
	S string `json:"s"`
}
