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
// +k8s:validation-gen-test-fixture=validateFalse

// This is a test package.
package embedded

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type T1 struct {
	TypeMeta int

	// NOTE: It's weird to have IntField in both, but Go allows it.
	T2  `json:",inline"`
	*T3 `json:",inline"`
}

type T2 struct {
	// +k8s:validateFalse="T2.IntField"
	IntField int `json:"intField"`
}

type T3 struct {
	// +k8s:validateFalse="T3.StringField"
	StringField string `json:"stringField"`

	// +k8s:validateFalse="T3.IntField"
	IntField int `json:"intField"`
}
