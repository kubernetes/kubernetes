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

// +k8s:validation-gen=TypeMeta
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme

// This is a test package.
package maxlength

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int

	// +k8s:maxLength=0
	Max0Field string `json:"max0Field"`

	// +k8s:maxLength=10
	Max10Field string `json:"max10Field"`

	// +k8s:maxLength=0
	Max0UnvalidatedTypedefField UnvalidatedStringType `json:"max0UnvalidatedTypedefField"`

	// +k8s:maxLength=10
	Max10UnvalidatedTypedefField UnvalidatedStringType `json:"max10UnvalidatedTypedefField"`

	// Note: no validation here
	Max0ValidatedTypedefField Max0Type `json:"max0ValidatedTypedefField"`

	// Note: no validation here
	Max10ValidatedTypedefField Max10Type `json:"max10ValidatedTypedefField"`
}

// Note: no validation here
type UnvalidatedStringType string

// +k8s:maxLength=0
type Max0Type string

// +k8s:maxLength=10
type Max10Type string
