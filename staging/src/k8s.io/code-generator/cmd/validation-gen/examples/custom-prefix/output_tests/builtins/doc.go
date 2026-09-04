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

// +xyz:validation-gen=TypesWithField=TypeMeta
// +xyz:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme

// This is a test package. It exercises the standard validation tags under
// the "xyz:" prefix.
package builtins

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int

	// +xyz:required
	// +xyz:minimum=1
	IntPtrField *int `json:"intPtrField"`

	// +xyz:optional
	// +xyz:maxLength=3
	StringField string `json:"stringField"`

	// +xyz:optional
	// +xyz:immutable
	ImmutableField string `json:"immutableField"`

	// +xyz:optional
	EnumField Enum `json:"enumField"`

	// +xyz:optional
	// +xyz:listType=map
	// +xyz:listMapKey=name
	// +xyz:eachVal=+xyz:subfield(name)=+xyz:maxLength=3
	ListMapField []Item `json:"listMapField"`
}

// +xyz:enum
type Enum string

const (
	EnumA Enum = "A"
	EnumB Enum = "B"
)

type Item struct {
	// +xyz:required
	Name string `json:"name"`
}
