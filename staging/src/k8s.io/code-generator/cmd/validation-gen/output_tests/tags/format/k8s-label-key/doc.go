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

// Package format is the internal version of the API.
// +k8s:validation:internal
package format

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int

	// +k8s:format=k8s-label-key
	LabelKeyField string `json:"labelKeyField"`

	// +k8s:format=k8s-label-key
	LabelKeyPtrField *string `json:"labelKeyPtrField"`

	// Note: no validation here
	LabelKeyTypedefField LabelKeyStringType `json:"labelKeyTypedefField"`
}

// +k8s:format=k8s-label-key
type LabelKeyStringType string
