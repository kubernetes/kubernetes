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
package mode

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type StrictUnion struct {
	TypeMeta int

	// +k8s:modeDiscriminator
	D1 string `json:"d1"`

	// +k8s:ifMode("A")=+k8s:required
	FieldA *string `json:"fieldA,omitempty"`

	// +k8s:ifMode("B")=+k8s:required
	FieldB *string `json:"fieldB,omitempty"`
}

type SharedField struct {
	TypeMeta int

	// +k8s:modeDiscriminator
	D1 string `json:"d1"`

	// Valid in A and B, implicitly forbidden in C.
	// +k8s:ifMode("A")=+k8s:optional
	// +k8s:ifMode("B")=+k8s:optional
	FieldA *string `json:"fieldA,omitempty"`
}

type ChainedValidation struct {
	TypeMeta int

	// +k8s:modeDiscriminator
	D1 string `json:"d1"`

	// In mode A, it is required AND must have maxLength 5.
	// +k8s:ifMode("A")=+k8s:required
	// +k8s:ifMode("A")=+k8s:maxLength=5
	FieldA *string `json:"fieldA,omitempty"`
}

type ImplicitForbidden struct {
	TypeMeta int

	// +k8s:modeDiscriminator
	D1 string `json:"d1"`

	// Field is only mentioned for mode A. Mode B should implicitly forbid it.
	// +k8s:ifMode("A")=+k8s:optional
	FieldA *string `json:"fieldA,omitempty"`
}

type NonStringDiscriminator struct {
	TypeMeta int

	// +k8s:modeDiscriminator(modality:"Bool")
	D1 bool `json:"d1"`

	// +k8s:ifMode(modality:"Bool", mode:"true")=+k8s:required
	FieldA *string `json:"fieldA,omitempty"`
}

type MultipleDiscriminators struct {
	TypeMeta int

	// +k8s:modeDiscriminator(modality:"D1")
	D1 string `json:"d1"`

	// +k8s:modeDiscriminator(modality:"D2")
	D2 string `json:"d2"`

	// +k8s:ifMode(modality:"D1", mode:"A")=+k8s:required
	FieldA *string `json:"fieldA,omitempty"`

	// +k8s:ifMode(modality:"D2", mode:"B")=+k8s:required
	FieldB *string `json:"fieldB,omitempty"`
}

type Collections struct {
	TypeMeta int

	// +k8s:modeDiscriminator
	D1 string `json:"d1"`

	// +k8s:ifMode("A")=+k8s:optional
	ListField []string `json:"listField,omitempty"`

	// +k8s:ifMode("A")=+k8s:optional
	MapField map[string]string `json:"mapField,omitempty"`
}

type TypeMeta int
