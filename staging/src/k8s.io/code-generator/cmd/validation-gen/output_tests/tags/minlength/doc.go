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

	// +k8s:minLength=1
	Min1Field string `json:"min1Field"`

	// +k8s:minLength=1
	Min1PtrField *string `json:"min1PtrField"`

	// +k8s:minLength=10
	Min10Field string `json:"min10Field"`

	// +k8s:minLength=10
	Min10PtrField *string `json:"min10PtrField"`

	// +k8s:minLength=1
	Min1UnvalidatedTypedefField UnvalidatedStringType `json:"min1UnvalidatedTypedefField"`

	// +k8s:minLength=1
	Min1UnvalidatedTypedefPtrField *UnvalidatedStringType `json:"min1UnvalidatedTypedefPtrField"`

	// +k8s:minLength=10
	Min10UnvalidatedTypedefField UnvalidatedStringType `json:"min10UnvalidatedTypedefField"`

	// +k8s:minLength=10
	Min10UnvalidatedTypedefPtrField *UnvalidatedStringType `json:"min10UnvalidatedTypedefPtrField"`

	// Note: no validation here
	Min1ValidatedTypedefField Min1Type `json:"min1ValidatedTypedefField"`

	// Note: no validation here
	Min1ValidatedTypedefPtrField *Min1Type `json:"min1ValidatedTypedefPtrField"`

	// Note: no validation here
	Min10ValidatedTypedefField Min10Type `json:"min10ValidatedTypedefField"`

	// Note: no validation here
	Min10ValidatedTypedefPtrField *Min10Type `json:"min10ValidatedTypedefPtrField"`
}

// Note: no validation here
type UnvalidatedStringType string

// +k8s:minLength=1
type Min1Type string

// +k8s:minLength=10
type Min10Type string
