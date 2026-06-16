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

// +k8s:validation-gen-nolint
package modes

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type AlphaStruct struct {
	TypeMeta int

	// +k8s:alpha=+k8s:modeDiscriminator
	D1 string `json:"d1"`

	// +k8s:alpha=+k8s:ifMode("A")=+k8s:required
	FieldA *string `json:"fieldA,omitempty"`

	// +k8s:alpha=+k8s:ifMode("B")=+k8s:required
	FieldB *string `json:"fieldB,omitempty"`
}

type BetaStruct struct {
	TypeMeta int

	// +k8s:beta=+k8s:modeDiscriminator
	D1 string `json:"d1"`

	// +k8s:beta=+k8s:ifMode("A")=+k8s:required
	FieldA *string `json:"fieldA,omitempty"`

	// +k8s:beta=+k8s:ifMode("B")=+k8s:required
	FieldB *string `json:"fieldB,omitempty"`
}

type MixedLevels struct {
	TypeMeta int

	// +k8s:modeDiscriminator
	Mode string `json:"mode"`

	// +k8s:alpha=+k8s:ifMode("A")=+k8s:required
	A *string `json:"a,omitempty"`

	// +k8s:beta=+k8s:ifMode("B")=+k8s:required
	B *string `json:"b,omitempty"`
}

type CrossLevels struct {
	TypeMeta int

	// +k8s:beta=+k8s:modeDiscriminator
	Kind string `json:"kind"`

	// +k8s:alpha=+k8s:ifMode("A")=+k8s:required
	A *string `json:"a,omitempty"`

	// +k8s:alpha=+k8s:ifMode("B")=+k8s:required
	B *string `json:"b,omitempty"`
}

type SameFieldMixed struct {
	TypeMeta int

	// +k8s:modeDiscriminator
	Mode string `json:"mode"`

	// +k8s:alpha=+k8s:ifMode("A")=+k8s:required
	// +k8s:beta=+k8s:ifMode("B")=+k8s:required
	Value *string `json:"value,omitempty"`
}

// SameValueMixedPayloads tests that multiple payload validations on the same
// discriminator value can have different stability levels.
type SameValueMixedPayloads struct {
	TypeMeta int

	// +k8s:modeDiscriminator
	Mode string `json:"mode"`

	// +k8s:alpha=+k8s:ifMode("A")=+k8s:required
	// +k8s:beta=+k8s:ifMode("A")=+k8s:minLength=3
	Value *string `json:"value,omitempty"`
}

type TypeMeta int
