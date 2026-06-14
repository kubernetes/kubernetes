/*
Copyright The Kubernetes Authors.

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

// Package shortcircuit contains test types for testing subfield short-circuit behavior.
// +k8s:validation-gen-nolint
package shortcircuit

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type TargetWithRequired struct {
	// +k8s:required
	Value *string `json:"value"`
}

type TargetWithImmutable struct {
	// +k8s:immutable
	Value string `json:"value"`
	Other string `json:"other"`
}

type ParentWithRequired struct {
	TypeMeta int `json:"typeMeta"`
	// +k8s:subfield(value)=+k8s:validateFalse="subfield ParentWithRequired.Field.Value"
	Field TargetWithRequired `json:"field"`
}

type ParentWithImmutable struct {
	TypeMeta int `json:"typeMeta"`
	// +k8s:subfield(value)=+k8s:validateFalse="subfield ParentWithImmutable.Field.Value"
	Field TargetWithImmutable `json:"field"`
}

type ParentWithOpaqueField struct {
	TypeMeta int `json:"typeMeta"`
	// +k8s:subfield(value)=+k8s:validateFalse="subfield ParentWithOpaqueField.Field.Value"
	// +k8s:opaqueType
	Field TargetWithRequired `json:"field"`
}

type ParentWithOpaqueImmutableField struct {
	TypeMeta int `json:"typeMeta"`
	// +k8s:subfield(value)=+k8s:validateFalse="subfield ParentWithOpaqueImmutableField.Field.Value"
	// +k8s:opaqueType
	Field TargetWithImmutable `json:"field"`
}

type ParentWithAlphaOpaqueField struct {
	TypeMeta int `json:"typeMeta"`
	// +k8s:subfield(value)=+k8s:validateFalse="subfield ParentWithAlphaOpaqueField.Field.Value"
	// +k8s:alpha=+k8s:opaqueType
	Field TargetWithRequired `json:"field"`
}

type ParentWithAlphaOpaqueImmutableField struct {
	TypeMeta int `json:"typeMeta"`
	// +k8s:subfield(value)=+k8s:validateFalse="subfield ParentWithAlphaOpaqueImmutableField.Field.Value"
	// +k8s:alpha=+k8s:opaqueType
	Field TargetWithImmutable `json:"field"`
}

// +k8s:opaqueType
type AliasOpaqueTargetWithRequired TargetWithRequired

type ParentWithOpaqueAlias struct {
	TypeMeta int `json:"typeMeta"`
	// +k8s:subfield(value)=+k8s:validateFalse="subfield ParentWithOpaqueAlias.Field.Value"
	Field AliasOpaqueTargetWithRequired `json:"field"`
}

// +k8s:opaqueType
type AliasOpaqueTargetWithImmutable TargetWithImmutable

type ParentWithOpaqueImmutableAlias struct {
	TypeMeta int `json:"typeMeta"`
	// +k8s:subfield(value)=+k8s:validateFalse="subfield ParentWithOpaqueImmutableAlias.Field.Value"
	Field AliasOpaqueTargetWithImmutable `json:"field"`
}

type ParentWithMultipleShortCircuit struct {
	TypeMeta int `json:"typeMeta"`
	// +k8s:required
	// +k8s:subfield(value)=+k8s:immutable
	// +k8s:subfield(value)=+k8s:validateFalse="subfield ParentWithMultipleShortCircuit.Field.Value"
	Field *TargetWithRequired `json:"field"`
}

type TargetWithOptional struct {
	// +k8s:optional
	Value *string `json:"value"`
}

type ParentWithOptional struct {
	TypeMeta int `json:"typeMeta"`
	// +k8s:subfield(value)=+k8s:validateFalse="subfield ParentWithOptional.Field.Value"
	Field TargetWithOptional `json:"field"`
}

type TargetWithForbidden struct {
	// +k8s:forbidden
	Value *string `json:"value"`
}

type ParentWithForbidden struct {
	TypeMeta int `json:"typeMeta"`
	// +k8s:subfield(value)=+k8s:validateFalse="subfield ParentWithForbidden.Field.Value"
	Field TargetWithForbidden `json:"field"`
}

type TargetWithUpdate struct {
	// +k8s:update=NoModify
	Value string `json:"value"`
}

type ParentWithUpdate struct {
	TypeMeta int `json:"typeMeta"`
	// +k8s:subfield(value)=+k8s:validateFalse="subfield ParentWithUpdate.Field.Value"
	Field TargetWithUpdate `json:"field"`
}

type TargetWithMaxItems struct {
	// +k8s:maxItems=2
	Value []string `json:"value"`
}

type ParentWithMaxItems struct {
	TypeMeta int `json:"typeMeta"`
	// +k8s:subfield(value)=+k8s:validateFalse="subfield ParentWithMaxItems.Field.Value"
	Field TargetWithMaxItems `json:"field"`
}

type TargetWithMaxProperties struct {
	// +k8s:maxProperties=2
	Value map[string]string `json:"value"`
}

type ParentWithMaxProperties struct {
	TypeMeta int `json:"typeMeta"`
	// +k8s:subfield(value)=+k8s:validateFalse="subfield ParentWithMaxProperties.Field.Value"
	Field TargetWithMaxProperties `json:"field"`
}
