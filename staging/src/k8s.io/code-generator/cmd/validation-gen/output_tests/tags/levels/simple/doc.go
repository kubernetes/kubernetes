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

package simple

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int

	// +k8s:alpha=+k8s:minimum=10
	IntField int `json:"intField"`

	// +k8s:beta=+k8s:minimum=10
	IntFieldBeta int `json:"intFieldBeta"`

	// +k8s:alpha=+k8s:maxLength=5
	StringField string `json:"stringField"`

	// +k8s:beta=+k8s:maxLength=5
	StringFieldBeta string `json:"stringFieldBeta"`

	// +k8s:alpha=+k8s:maxItems=2
	SliceField []string `json:"sliceField"`

	// +k8s:beta=+k8s:maxItems=2
	SliceFieldBeta []string `json:"sliceFieldBeta"`

	// +k8s:alpha=+k8s:format=k8s-uuid
	UUIDField string `json:"uuidField"`

	// +k8s:beta=+k8s:format=k8s-uuid
	UUIDFieldBeta string `json:"uuidFieldBeta"`

	// +k8s:alpha=+k8s:immutable
	ImmutableField string `json:"immutableField"`

	// +k8s:beta=+k8s:immutable
	ImmutableFieldBeta string `json:"immutableFieldBeta"`
}

type SpecialValidationStruct struct {
	TypeMeta int

	// +k8s:alpha=+k8s:neq=5
	NEQField int `json:"neqField"`

	// +k8s:beta=+k8s:neq=5
	NEQFieldBeta int `json:"neqFieldBeta"`

	// +k8s:alpha=+k8s:forbidden
	ForbiddenField *string `json:"forbiddenField"`

	// +k8s:beta=+k8s:forbidden
	ForbiddenFieldBeta *string `json:"forbiddenFieldBeta"`

	// +k8s:alpha=+k8s:update=NoModify
	UpdateField string `json:"updateField"`

	// +k8s:beta=+k8s:update=NoModify
	UpdateFieldBeta string `json:"updateFieldBeta"`
}

type StructWithValidateFalse struct {
	TypeMeta int

	// +k8s:alpha=+k8s:validateFalse="always fails"
	ValidateFalse *string `json:"validateFalse"`

	// +k8s:beta=+k8s:validateFalse="always fails"
	ValidateFalseBeta *string `json:"validateFalseBeta"`
}
