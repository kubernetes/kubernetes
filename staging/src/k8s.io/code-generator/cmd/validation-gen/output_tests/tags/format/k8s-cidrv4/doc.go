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
// +k8s:validation-gen-nolint
package k8scidrv4

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

// MyType is a struct that contains a field with the k8s-cidrv4 format.
// +k8s:validation:Required
type MyType struct {
	TypeMeta int
	// +k8s:optional
	// +k8s:format=k8s-cidrv4
	CIDRv4Field string `json:"cidrv4Field"`
	// +k8s:optional
	// +k8s:format=k8s-cidrv4
	CIDRv4PtrField *string `json:"cidrv4PtrField"`
	// Note: no validation here
	CIDRv4TypedefField CIDRv4StringType `json:"cidrv4TypedefField"`
}

// +k8s:format=k8s-cidrv4
type CIDRv4StringType string
