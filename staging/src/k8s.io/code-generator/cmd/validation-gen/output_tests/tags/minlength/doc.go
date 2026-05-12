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

// This is a test package.
package minlength

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int

	// +k8s:alpha=+k8s:minLength=0
	// +k8s:optional
	Min0PtrField *string `json:"min0PtrField"`

	// +k8s:alpha=+k8s:minLength=2
	// +k8s:optional
	Min2Field string `json:"min2Field"`

	// +k8s:alpha=+k8s:minLength=2
	// +k8s:optional
	Min2PtrField *string `json:"min2PtrField"`

	// +k8s:alpha=+k8s:minLength=10
	// +k8s:optional
	Min10Field string `json:"min10Field"`

	// +k8s:alpha=+k8s:minLength=10
	// +k8s:optional
	Min10PtrField *string `json:"min10PtrField"`

	// +k8s:alpha=+k8s:minLength=2
	// +k8s:optional
	Min2UnvalidatedTypedefField UnvalidatedStringType `json:"min2UnvalidatedTypedefField"`

	// +k8s:alpha=+k8s:minLength=2
	// +k8s:optional
	Min2UnvalidatedTypedefPtrField *UnvalidatedStringType `json:"min2UnvalidatedTypedefPtrField"`

	// +k8s:alpha=+k8s:minLength=10
	// +k8s:optional
	Min10UnvalidatedTypedefField UnvalidatedStringType `json:"min10UnvalidatedTypedefField"`

	// +k8s:alpha=+k8s:minLength=10
	// +k8s:optional
	Min10UnvalidatedTypedefPtrField *UnvalidatedStringType `json:"min10UnvalidatedTypedefPtrField"`

	// Note: no minlength validation here
	// +k8s:optional
	Min2ValidatedTypedefField Min2Type `json:"min2ValidatedTypedefField"`

	// Note: no minlength validation here
	// +k8s:optional
	Min2ValidatedTypedefPtrField *Min2Type `json:"min2ValidatedTypedefPtrField"`

	// Note: no minlength validation here
	// +k8s:optional
	Min10ValidatedTypedefField Min10Type `json:"min10ValidatedTypedefField"`

	// Note: no minlength validation here
	// +k8s:optional
	Min10ValidatedTypedefPtrField *Min10Type `json:"min10ValidatedTypedefPtrField"`

	// +k8s:alpha=+k8s:minLength=2
	// +k8s:optional
	Min2UnvalidatedStringAliasField UnvalidatedStringAlias `json:"min2UnvalidatedStringAliasField"`

	// +k8s:alpha=+k8s:minLength=2
	// +k8s:optional
	Min2UnvalidatedStringAliasPtrField *UnvalidatedStringAlias `json:"min2UnvalidatedStringAliasPtrField"`
}

// Note: no validation here
type UnvalidatedStringType string

// Note: no validation here
type UnvalidatedStringAlias = string

// Tests that min length markers on typedefs
// appropriately propagate to fields that use this type
// +k8s:alpha=+k8s:minLength=2
type Min2Type string

// Tests that min length markers on typedefs
// appropriately propagate to fields that use this type
// +k8s:alpha=+k8s:minLength=10
type Min10Type string
