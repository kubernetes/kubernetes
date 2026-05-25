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
// +k8s:validation-gen-nolint
package typedeftoslice

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

// Note: no validation here
type UnvalidatedType []int

// +k8s:minItems=0
type Min0Type []int

// +k8s:minItems=10
type Min10Type []int

// Note: no validation here
type UnvalidatedPtrType []*int

type SliceType []int

// +k8s:minItems=0
type Min0TypedefType SliceType

// +k8s:minItems=10
type Min10TypedefType SliceType

type Struct struct {
	TypeMeta int

	UnvalidatedField UnvalidatedType `json:"unvalidatedField"`

	Min0Field Min0Type `json:"min0Field"`

	Min10Field Min10Type `json:"min10Field"`

	Min0TypedefField Min0TypedefType `json:"min0TypedefField"`

	Min10TypedefField Min10TypedefType `json:"min10TypedefField"`
}
