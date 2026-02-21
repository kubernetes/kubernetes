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

package unions

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int

	// +k8s:alpha=+k8s:unionDiscriminator
	D D `json:"d"`

	// +k8s:alpha=+k8s:unionMember
	// +k8s:optional
	M1 *M1 `json:"m1"`

	// +k8s:alpha(since:"1.35")=+k8s:unionMember
	// +k8s:optional
	M2 *M2 `json:"m2"`
}

type UnionStructBeta struct {
	TypeMeta int

	// +k8s:beta=+k8s:unionDiscriminator
	DBeta BetaD `json:"dBeta"`

	// +k8s:beta=+k8s:unionMember
	// +k8s:optional
	M1Beta *BetaM1 `json:"m1Beta"`

	// +k8s:beta(since:"1.35")=+k8s:unionMember
	// +k8s:optional
	M2Beta *BetaM2 `json:"m2Beta"`
}

type MyStruct struct {
	TypeMeta int

	// +k8s:alpha=+k8s:zeroOrOneOfMember
	// +k8s:optional
	Z1 *Z1 `json:"z1"`

	// +k8s:alpha=+k8s:zeroOrOneOfMember
	// +k8s:optional
	Z2 *Z2 `json:"z2"`
}

type MyStructBeta struct {
	TypeMeta int

	// +k8s:beta=+k8s:zeroOrOneOfMember
	// +k8s:optional
	Z1Beta *BetaZ1 `json:"z1Beta"`

	// +k8s:beta=+k8s:zeroOrOneOfMember
	// +k8s:optional
	Z2Beta *BetaZ2 `json:"z2Beta"`
}

type D string

const (
	DM1 D = "M1"
	DM2 D = "M2"
)

type M1 struct{}
type M2 struct{}

type BetaD string

const (
	BetaDM1 BetaD = "M1Beta"
	BetaDM2 BetaD = "M2Beta"
)

type BetaM1 struct{}
type BetaM2 struct{}

type Z1 struct{}
type Z2 struct{}

type BetaZ1 struct{}
type BetaZ2 struct{}
