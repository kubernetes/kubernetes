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

// +k8s:validation-gen=*
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme
// +k8s:validation-gen-test-fixture=validateFalse

// Package publicprivate is a test package.
//
//nolint:unused,govet // govet disables structtag check, which checks for use of tags on private fields
package publicprivate

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type T1 struct {
	// +k8s:validateFalse="field T1.Public"
	Public string `json:"public"`

	// +k8s:validateFalse="field T1.private"
	private string `json:"private"`
}

type private struct {
	// +k8s:validateFalse="field private.Public"
	Public string `json:"public"`

	// +k8s:validateFalse="field private.private"
	private string `json:"private"`
}
