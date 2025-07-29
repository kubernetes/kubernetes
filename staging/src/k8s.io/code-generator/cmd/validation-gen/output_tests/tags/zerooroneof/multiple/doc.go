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
package multiple

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

// Two non-discriminated zero-or-one-of unions in the same struct
type Struct struct {
	TypeMeta int

	NonUnionField string `json:"nonUnionField"`

	// +k8s:zeroOrOneOfMember(union: "union1")
	// +k8s:optional
	U1M1 *M1 `json:"u1m1"`

	// +k8s:zeroOrOneOfMember(union: "union1")
	// +k8s:optional
	U1M2 *M2 `json:"u1m2"`

	// +k8s:zeroOrOneOfMember(union: "union2")
	// +k8s:optional
	U2M1 *M1 `json:"u2m1"`

	// +k8s:zeroOrOneOfMember(union: "union2")
	// +k8s:optional
	U2M2 *M2 `json:"u2m2"`
}

type M1 struct{}

type M2 struct{}
