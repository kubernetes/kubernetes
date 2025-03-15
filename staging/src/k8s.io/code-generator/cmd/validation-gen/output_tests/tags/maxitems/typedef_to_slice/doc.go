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
package typedeftoslice

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

// Note: no validation here
type UnvalidatedType []int

// +k8s:maxItems=0
type Max0Type []int

// +k8s:maxItems=10
type Max10Type []int

// Note: no validation here
type UnvalidatedPtrType []*int

type SliceType []int

// +k8s:maxItems=0
type Max0TypedefType SliceType

// +k8s:maxItems=10
type Max10TypedefType SliceType

type Struct struct {
	TypeMeta int

	UnvalidatedField UnvalidatedType `json:"unvalidatedField"`

	Max0Field Max0Type `json:"max0Field"`

	Max10Field Max10Type `json:"max10Field"`

	Max0TypedefField Max0TypedefType `json:"max0TypedefField"`

	Max10TypedefField Max10TypedefType `json:"max10TypedefField"`
}
